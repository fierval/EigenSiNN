#pragma once

#define MAX_ELEM 1e9

#include "ops/linearops.hpp"
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


  template<typename Scalar, typename Device_ = DefaultDevice>
  class Linear : public LayerBase<Scalar, Device_> {

  public:
    Linear(int _in_dim, int _out_dim, Dispatcher<Device_>& _device =  LayerBase::default_dispatcher) :
      LayerBase<Scalar, Device_>(_device),
      layer_output(1, _out_dim),
      layer_grad_loss_by_input(1, _in_dim),
      layer_grad_loss_by_weight(_in_dim, _out_dim),
      weights(_in_dim, _out_dim),
      loss_by_bias_derivative(_out_dim),
      in_dim(_in_dim),
      out_dim(_out_dim),
      broadcast_bias_dim({ 1, 1 }),
      bias(_out_dim)   {
        
      weights_gpu = layer_output_gpu = layer_grad_loss_by_weight_gpu = layer_grad_loss_by_input_gpu = nullptr;
      bias_gpu = loss_by_bias_derivative_gpu = nullptr;
    }

    // prev_layer_out: X[l-1], dim: [N, D]
    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      int batch_size = broadcast_bias_dim[0] = prev_layer.get_out_dims()[0];

      if (are_dims_unset(prev_layer.get_out_dims())) {
        std::vector<Index> _out_dims{ batch_size, out_dim };
        set_dims(prev_layer.get_out_dims(), _out_dims);

        layer_output.resize(batch_size, layer_output.dimension(1));
        layer_grad_loss_by_input.resize(batch_size, layer_grad_loss_by_input.dimension(1));
      }

      if (should_move_to_gpu()) {
        weights_gpu = to_gpu_tensor(weights);
      }

      // dims: [N, D] * [D, M] -> [N, M]
      ProductDims prod_dims = { IndexPair<int>(1, 0) };
      TensorMap<Tensor<Scalar, 2>> prev_layer_tensor(prev_layer.get_output(), vector2array<2>(in_dims));

      layer_output.device(dispatcher.get_device()) = prev_layer_tensor.contract(weights, prod_dims);

      // bias: [1, M]
      layer_output.device(dispatcher.get_device()) += bias.reshape(array<Index, 2>{ 1, bias.dimension(0) }).broadcast(broadcast_bias_dim);
    }

    // next_layer_grad: delta[l+1] = dL/dX[l+1], dims: [N, M] (same as X[l+1])
    // when we are feeding backward from the loss function
    void backward(LayerBase<Scalar, Device_>& prev_layer_any, Scalar * next_layer_grad_any) override {

      TensorMap<Tensor<Scalar, 2>> prev_layer(prev_layer_any.get_output(), vector2array< 2>(in_dims));
      TensorMap<Tensor<Scalar, 2>> next_layer_grad(next_layer_grad_any, vector2array< 2>(out_dims));

      // this will be fed to the previous backprop layer as the delta parameter
      // dL/dX = dim delta[l+1] * w.T: [N, M] * [M, D] -> [N, D] (same as X[l-1])
      ProductDims prod_dims = { IndexPair<int>(1, 1) };
      layer_grad_loss_by_input.device(dispatcher.get_device()) = next_layer_grad.contract(weights, prod_dims);

      //dl/dW = dim X[l].T * delta[l+1]: [D, N] * [N, M] -> [D, M], same as W
      prod_dims = { IndexPair<int>(0, 0) };
      layer_grad_loss_by_weight.device(dispatcher.get_device()) = prev_layer.contract(next_layer_grad, prod_dims);

      //db: dL/dY * dY/db = sum_j(dL/dY_j) dim: [1, M], same as bias
      loss_by_bias_derivative.device(dispatcher.get_device()) = next_layer_grad.sum(reduce_bias_dim);
    }

    void init(const Tensor<Scalar, 2>& _weights) {
      init();
      weights = _weights;
    }

    void init(const Tensor<Scalar, 2>& _weights, const Tensor<Scalar, 1>& _bias) {

      init(_weights);
      bias = _bias;
    }

    // TODO: actual initialization needed
    void init() override {
      if (in_dim <= 0 || out_dim <= 0 || in_dim > MAX_ELEM || out_dim > MAX_ELEM) {
        throw std::invalid_argument("inappropriate dimensions");
      }

      set_bias_dims(std::vector<Index> {out_dim});
      set_weight_dims(std::vector<Index>{in_dim, out_dim});

      //weights of dimension (D, M)
      weights = generate_xavier<Scalar, 2>(vector2array< 2>(weight_dims), dispatcher.get_device());
      bias = bias.setZero();
    }

    // this will be fed to compute dL/dW[l-1]
    // it is dL/dX[l]
     Scalar * get_loss_by_input_derivative() {
      return layer_grad_loss_by_input.data();
    }

    // feed to optimizer
     Scalar * get_loss_by_weights_derivative() override {
      return layer_grad_loss_by_weight.data();
    }

     Scalar * get_loss_by_bias_derivative() override {
      return loss_by_bias_derivative.data();
    }

     Scalar * get_output() override {
      return layer_output.data();
    }

     Scalar * get_weights() override {
      return weights.data();
    }

     Scalar * get_bias() override {
      return bias.data();
    }

  private:

    Tensor<Scalar, 2> weights;
    Tensor<Scalar, 2> layer_output, layer_grad_loss_by_weight, layer_grad_loss_by_input;
    Tensor<Scalar, 1> bias, loss_by_bias_derivative;

    // GPU support
    TensorMap<Tensor<Scalar, 2>> *weights_gpu, *layer_output_gpu, *layer_grad_loss_by_weight_gpu, *layer_grad_loss_by_input_gpu;
    TensorMap<Tensor<Scalar, 1>> *bias_gpu, *loss_by_bias_derivative_gpu;

    const int in_dim, out_dim;
    array<int, 2> broadcast_bias_dim;
    const array<int, 1> reduce_bias_dim = { 0 };
  };
}