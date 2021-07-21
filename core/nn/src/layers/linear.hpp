#pragma once

#define MAX_ELEM 1e9

#include "ops/conversions.hpp"
#include "ops/initializations.hpp"
#include "layer_base.hpp"

#include <onnx/op_defs.h>

using namespace Eigen;

/*
Fully connected layer X[l]
_in_dim - input dimension (D)
-out_dim - output dimension (M)
*/
namespace EigenSinn {


  template<typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class Linear : public LayerBase<Scalar, Device_> {

  public:
    Linear(int _in_dim, int _out_dim) :
      LayerBase<Scalar, Device_>(gemm_op),
      layer_grad_loss_by_weight(_in_dim, _out_dim),
      weights(_in_dim, _out_dim),
      loss_by_bias_derivative(_out_dim),
      in_dim(_in_dim),
      out_dim(_out_dim),
      broadcast_bias_dim({ 0, 1 }),
      bias(_out_dim) {

    }

    // prev_layer_out: X[l-1], dim: [N, D]
    void forward(PtrTensorAdapter<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, 2, Device_, Layout > prev_layer_tensor(prev_layer);

      int batch_size = prev_layer_tensor->dimension(0);

      // resize the inputs for the right dimensions
      if (!layer_output || broadcast_bias_dim[0] != batch_size) {
        broadcast_bias_dim[0] = batch_size;
        layer_output.resize(batch_size, out_dim);
        layer_grad_loss_by_input.resize(batch_size, in_dim);
      }

      // dims: [N, D] * [D, M] -> [N, M]
      ProductDims prod_dims = { IndexPair<int>(1, 0) };
      
      layer_output.view() = prev_layer_tensor->contract(*weights, prod_dims);

      // bias: [1, M]
      layer_output.view() += bias->reshape(array<Index, 2>{ 1, bias->dimension(0) }).broadcast(broadcast_bias_dim);
    }

    // next_layer_grad: delta[l+1] = dL/dX[l+1], dims: [N, M] (same as X[l+1])
    // when we are feeding backward from the loss function
    void backward(PtrTensorAdapter<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, 2, Device_, Layout> prev_layer(prev_layer_any);
      DeviceTensor<Scalar, 2, Device_, Layout> next_layer_grad(next_layer_grad_any);

      // this will be fed to the previous backprop layer as the delta parameter
      // dL/dX = dim delta[l+1] * w.T: [N, M] * [M, D] -> [N, D] (same as X[l-1])
      ProductDims prod_dims = { IndexPair<int>(1, 1) };
      layer_grad_loss_by_input.view() = next_layer_grad->contract(*weights, prod_dims);

      //dl/dW = dim X[l].T * delta[l+1]: [D, N] * [N, M] -> [D, M], same as W
      prod_dims = { IndexPair<int>(0, 0) };
      layer_grad_loss_by_weight.view() = prev_layer->contract(*next_layer_grad, prod_dims);

      //db: dL/dY * dY/db = sum_j(dL/dY_j) dim: [1, M], same as bias
      loss_by_bias_derivative.view() = next_layer_grad->sum(reduce_bias_dim);
    }

    void init(Tensor<Scalar, 2, Layout>& _weights) {
      init();
      weights.set_from_host(_weights);
    }

    void init(Tensor<Scalar, 2, Layout>& _weights, Tensor<Scalar, 1, Layout>& _bias) {

      init(_weights);
      bias.set_from_host(_bias);
    }

    // TODO: actual initialization needed
    void init() override {
      if (in_dim <= 0 || out_dim <= 0 || in_dim > MAX_ELEM || out_dim > MAX_ELEM) {
        throw std::invalid_argument("inappropriate dimensions");
      }

      //weights of dimension (D, M)
      // Wrapping the pointer and moving it to the tensor keeping in mind the GPU device
      weights = generate_xavier<Scalar, 2, Layout, Device_>(weights.dimensions());
      bias.setZero();

    }

    // this will be fed to compute dL/dW[l-1]
    // it is dL/dX[l]
    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() {
      return layer_grad_loss_by_input.raw();
    }

    // feed to optimizer
    PtrTensorAdapter<Scalar, Device_> get_loss_by_weights_derivative() override {
      return layer_grad_loss_by_weight.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_bias_derivative() override {
      return loss_by_bias_derivative.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_weights() override {
      return weights.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_bias() override {
      return bias.raw();
    }

    void set_weights(PtrTensorAdapter<Scalar, Device_>& v) override {
      weights = DeviceTensor<Scalar, 2, Device_, Layout>(v);
    }

    void set_bias(PtrTensorAdapter<Scalar, Device_>& v) override {
      bias = DeviceTensor<Scalar, 1, Device_, Layout>(v);
    }

    const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override { 

      // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md
      // 1. Save initializers (raw data in row major format)
      weights.save_onnx_initializer(model);
      bias.save_onnx_initializer(model);

      std::vector<std::string> names{ input_name, weights.get_onnx_input_name(model), bias.get_onnx_input_name(model) };
      onnx::NodeProto* node = model.add_graph_node(get_op_name(), names);

      //TODO: single output
      const std::string& out_name = node->output().Get(0);

      // 2. create attributes
      auto alpha_attr = node->add_attribute();
      alpha_attr->set_name("alpha");
      alpha_attr->set_f(1);
      alpha_attr->set_type(onnx::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT);

      auto beta_attr = node->add_attribute();
      beta_attr->set_name("beta");
      beta_attr->set_f(1);
      beta_attr->set_type(onnx::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT);

      // return output to pass as input to next node in graph
      return out_name;

    }

    inline void load_onnx_data(EigenModel& model, std::vector<std::string>& inputs) override {

      std::vector<std::vector<Index>> dimensions;
      std::vector<onnx::TensorProto> values;

      std::tie(values, dimensions) = model.get_input_data_and_dimensions<Scalar>(inputs);

      weights.set_from_host(model.get_input_data<Scalar>(values[0]), vec2dims<2>(dimensions[0]));
      bias.set_from_host(model.get_input_data<Scalar>(values[1]), vec2dims<1>(dimensions[1]));

      weights.set_node_input_name(inputs[1]);
      bias.set_node_input_name(inputs[2]);
    }

    const std::vector<Index> onnx_out_dims() override {
      return layer_output.vec_dims();
    }

  private:

    DeviceTensor<Scalar, 2, Device_, Layout> weights;
    DeviceTensor<Scalar, 2, Device_, Layout> layer_output, layer_grad_loss_by_weight, layer_grad_loss_by_input;
    DeviceTensor<Scalar, 1, Device_, Layout> bias, loss_by_bias_derivative;

    const int in_dim, out_dim;
    array<int, 2> broadcast_bias_dim;
    const array<int, 1> reduce_bias_dim = { 0 };
  };
}