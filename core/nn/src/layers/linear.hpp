#pragma once

#define MAX_ELEM 1e9

#include "ops/linearops.hpp"
#include "ops/initializations.hpp"
#include "layer_base.hpp"

using namespace Eigen;

/*
Fully connected layer X[l]
_batch_size - size of the incoming minibatch (N)
_in_dim - input dimension (D)
-out_dim - output dimension (M)
*/
namespace EigenSinn {


  template<typename Scalar>
  class Linear : LayerBase {

  public:
    Linear(int _batch_size, int _in_dim, int _out_dim) :
      batch_size(_batch_size),
      in_dim(_in_dim),
      out_dim(_out_dim),
      broadcast_bias_dim({ _batch_size, 1 }),
      bias(_out_dim)   {

    }

    // prev_layer_out: X[l-1], dim: [N, D]
    void forward(std::any prev_layer) override {

      // dims: [N, D] * [D, M] -> [N, M]
      ProductDims prod_dims = { IndexPair<int>(1, 0) };
      layer_output = std::any_cast<Tensor<Scalar, 2>&>(prev_layer).contract(weights, prod_dims);

      // bias: [1, M]
      layer_output += bias.reshape(array<Index, 2>{ 1, bias.dimension(0) }).broadcast(broadcast_bias_dim);
    }

    // next_layer_grad: delta[l+1] = dL/dX[l+1], dims: [N, M] (same as X[l+1])
    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

      Tensor<Scalar, 2> prev_layer = std::any_cast<Tensor<Scalar, 2>&>(prev_layer_any);
      Tensor<Scalar, 2> next_layer_grad = std::any_cast<Tensor<Scalar, 2>&>(next_layer_grad_any);

      // this will be fed to the previous backprop layer as the delta parameter
      // dL/dX = dim delta[l+1] * w.T: [N, M] * [M, D] -> [N, D] (same as X[l-1])
      ProductDims prod_dims = { IndexPair<int>(1, 1) };
      layer_grad_loss_by_input = next_layer_grad.contract(weights, prod_dims);

      //dl/dW = dim X[l].T * delta[l+1]: [D, N] * [N, M] -> [D, M], same as W
      prod_dims = { IndexPair<int>(0, 0) };
      layer_grad_loss_by_weight = prev_layer.contract(next_layer_grad, prod_dims);

      //db: dL/dY * dY/db = sum_j(dL/dY_j) dim: [1, M], same as bias
      loss_by_bias_derivative = next_layer_grad.sum(reduce_bias_dim);
    }

    void init(const Tensor<Scalar, 2>& _weights) {
      weights = _weights;
      bias.setZero();
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

      //weights of dimension (D, M)
      weights = generate_xavier<Scalar, 2>(array<Index, 2>{in_dim, out_dim});
      bias = bias.setZero();
    }

    // this will be fed to compute dL/dW[l-1]
    // it is dL/dX[l]
    const std::any get_loss_by_input_derivative() {
      return layer_grad_loss_by_input;
    }

    // feed to optimizer
    const std::any get_loss_by_weights_derivative() override {
      return layer_grad_loss_by_weight;
    }

    const std::any get_loss_by_bias_derivative() override {
      return loss_by_bias_derivative;
    }

    const std::any get_output() override {
      return layer_output;
    }

    const std::any get_weights() override {
      return weights;
    }

    const std::any get_bias() {
      return bias;
    }

  private:

    Tensor<Scalar, 2> weights;
    Tensor<Scalar, 2> layer_output, layer_grad_loss_by_weight, layer_grad_loss_by_input;
    Tensor<Scalar, 1> bias, loss_by_bias_derivative;

    const int in_dim, out_dim, batch_size;
    const array<int, 2> broadcast_bias_dim;
    const array<int, 1> reduce_bias_dim = { 0 };
  };
}