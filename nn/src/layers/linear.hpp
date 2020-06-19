#pragma once

#define MAX_ELEM 1e9

#include "ops/linearops.hpp"
#include <stdexcept>
#include <type_traits>
#include "Random.h"
#include <unsupported\Eigen\CXX11\src\Tensor\TensorRandom.h>
#include "layer_base.hpp"

using namespace Eigen;

/*
Fully connected layer X[l]
_batch_size - size of the incoming minibatch (N)
_in_dim - input dimension (D)
-out_dim - output dimension (M)
*/
namespace EigenSinn {

  typedef array<IndexPair<int>, 1> ProductDims;

  class Linear : LayerBase{

  public:
    Linear(Index _batch_size, Index _in_dim, Index _out_dim, bool _use_bias = false) :
      batch_size(_batch_size),
      in_dim(_in_dim),
      out_dim(_out_dim),
      use_bias(_use_bias) {

      biased_dim = use_bias ? in_dim + 1 : in_dim;
    }

    // prev_layer_out: X[l-1], dim: [N, D]
    void forward(std::any prev_layer) override {

      // dims: [N, D] * [D, M] -> [N, M]

      ProductDims prod_dims = { IndexPair<int>(1, 0) };
      layer_output = std::any_cast<LinearTensor&>(prev_layer).contract(weights, prod_dims);

      // we assume that bias adjustment is done at the end of the forward pass
      // since weights will include a bias dimension at initialization, we don't need
      // to worry about biases anymore
      // (X,1) * W dim: [N, D+1] * [D + 1, M]
      if (use_bias) {
        layer_output = adjust_linear_bias(layer_output);
      }
    }

    // next_layer_grad: delta[l+1] = dL/dX[l+1], dims: [N, M] (same as X[l+1])
    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override{

      LinearTensor prev_layer = std::any_cast<LinearTensor&>(prev_layer_any);
      LinearTensor next_layer_grad = std::any_cast<LinearTensor&>(next_layer_grad_any);

      // this will be fed to the previous backprop layer as the delta parameter
      // dL/dX = dim delta[l+1] * w.T: [N, M] * [M, D] -> [N, D] (same as X[l-1])
      ProductDims prod_dims = { IndexPair<int>(1, 1) };
      layer_grad_loss_by_input = next_layer_grad.contract(weights, prod_dims);

      //dl/dW = dim X[l].T * delta[l+1]: [D, N] * [N, M] -> [D, M], same as W
      prod_dims = { IndexPair<int>(0, 0) };
      layer_grad_loss_by_weight = prev_layer.contract(next_layer_grad, prod_dims);
    }

    void init(RNG rng, double mu, double sigma, bool debug = false) {
      if (in_dim <= 0 || out_dim <= 0 || in_dim > MAX_ELEM || out_dim > MAX_ELEM) {
        throw std::invalid_argument("inappropriate dimensions");
      }

      //weights of dimension (D, M)
      weights.resize(biased_dim, out_dim);
      if (debug) {
        init_debug();
        return;
      }
      
      weights.setRandom<internal::NormalRandomGenerator<float>>();
      //set_normal_random(weights.data(), static_cast<int>(weights.size()), rng, mu, sigma);
    }

    // this will be fed to compute dL/dW[l-1]
    // it is dL/dX[l]
    LinearTensor& get_loss_by_input_derivative() {
      return layer_grad_loss_by_input;
    }

    // feed to optimizer
    LinearTensor& get_loss_by_weights_derivative() {
      return layer_grad_loss_by_weight;
    }

    LinearTensor& get_output() {
      return layer_output;
    }

    LinearTensor& get_weights() {
      return weights;
    }

  private:

    // init weights to 1 for debugging
    void init_debug() {
      weights.setConstant(1);
    }

    LinearTensor weights;
    LinearTensor layer_output, layer_grad_loss_by_weight, layer_grad_loss_by_input;

    const bool use_bias;
    const Index in_dim, out_dim, batch_size;
    Index biased_dim;
  };
}