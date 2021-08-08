#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>
#include <onnx/op_defs.h>

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class Dropout : public LayerBase<Scalar, Device_> {
  public:

    Dropout(float _prob = 0.5)
      : LayerBase<Scalar, Device_>(OnnxOpNames::dropout_op)
      , prob(_prob)
      , is_training(true)
      , inited(false) {
    }

    void init(const DeviceTensor<Scalar, Rank, Device_, Layout>& x)  {

      layer_gradient.resize(x.dimensions());
      layer_output.resize(x.dimensions());
      mask.resize(x.dimensions());
      rands.resize(x.dimensions());

      if_tensor.resize(x.dimensions());
      
      then_tensor.resize(x.dimensions());
      then_tensor.setConstant(1. / (1. - prob));

      else_tensor.resize(x.dimensions());
      else_tensor.setZero();

      prob_tensor.resize(x.dimensions());
      prob_tensor.setConstant(prob);
    }

    void forward(PtrTensorAdaptor<Scalar, Device_>& prev_layer) override {
      
      if (!is_training) { return; }

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer);

      if (!inited) {
        inited = true;
        init(x);
      }

      // generate random uniform values
      rands.template setRandom<internal::UniformRandomGenerator<float>>();
            
      // create condition
      if_tensor.view() = *rands >= *prob_tensor;
      mask.view() = if_tensor->select(*then_tensor, *else_tensor);

      layer_output.view() = *mask * *x;
    }

    // for derivations
    void backward(PtrTensorAdaptor<Scalar, Device_>& prev_layer_any, PtrTensorAdaptor<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> next_layer_grad(next_layer_grad_any);

      if (!is_training) { return; }

      layer_gradient.view() = *mask * *next_layer_grad;
    }

    PtrTensorAdaptor<Scalar, Device_> get_output() override {
      return layer_output.raw();
    }

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative() {
      return layer_gradient.raw();
    }

    void set_training(bool _is_training) { 
      is_training = _is_training;
    }

    bool get_training() {
      return is_training;
    }

    const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override {

      // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#Dropout
      // Dropout spec saves everything as input
      // So we need to wrap scalar values in tensors
      DeviceTensor<float, 0> prob_tensor;
      prob_tensor.setConstant(prob);

      // 1. Add initializers
      prob_tensor.save_onnx_initializer(model);

      // 2. Add inputs
      auto* node = model.add_graph_node(get_layer_name(), get_op_name(),
        std::vector<std::string>{input_name, prob_tensor.get_onnx_input_name(model)});

      // save rank, not part of ONNX but necessary for loading
      model.add_attr(node, "rank", Rank);

      const std::string out_name = node->output().Get(0);
      return out_name;
    }


    // in the order they appear in the ONNX file
    inline void load_onnx_data(EigenModel& model, std::vector<std::string>& inputs) override {

      std::vector<onnx::TensorProto> values;
      std::vector<std::vector<Index>> dimensions;

      // just one value and dims = 0 
      std::tie(values, dimensions) = model.get_input_data_and_dimensions<Scalar>(inputs);

      // recover probability
      prob = *model.get_input_data<Scalar>(values[0]);

    }

    const std::vector<Index> onnx_out_dims() override {
      return layer_output.vec_dims();
    }

  private:
    DeviceTensor<Scalar, Rank, Device_, Layout> mask;
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_gradient;
    DeviceTensor<bool, Rank, Device_, Layout> if_tensor;
    DeviceTensor<float, Rank, Device_, Layout> rands, then_tensor, else_tensor, prob_tensor;
    bool is_training, inited;
    float prob;
  };

}