#pragma once

#define MAX_ELEM 1e9

#include "ops/conversions.hpp"
#include "ops/initializations.hpp"
#include "layer_base.hpp"

using namespace Eigen;

/*
Fully connected layer X[l]
_in_dim - input dimension (D)
-out_dim - output dimension (M)
*/
namespace EigenSinn {


  template<typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class Linear : public LayerBase<Scalar, Device_> {

  public:
    Linear(int _in_dim, int _out_dim) :
      layer_grad_loss_by_weight(_in_dim, _out_dim),
      weights(_in_dim, _out_dim),
      loss_by_bias_derivative(_out_dim),
      in_dim(_in_dim),
      out_dim(_out_dim),
      broadcast_bias_dim({ 0, 1 }),
      bias(_out_dim) {

    }

    // prev_layer_out: X[l-1], dim: [N, D]
    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, 2, Device_, Layout > prev_layer_tensor(prev_layer.get_output());

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
    void backward(LayerBase<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, 2, Device_, Layout> prev_layer(prev_layer_any.get_output());
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

    void init(Tensor<Scalar, 2>& _weights) {
      init();
      weights.set_from_host(_weights);
    }

    void init(Tensor<Scalar, 2>& _weights, Tensor<Scalar, 1>& _bias) {

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

    void set_bias(PtrTensorAdapter<Scalar, Device_>& v) override{
      bias = DeviceTensor<Scalar, 1, Device_, Layout>(v);
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