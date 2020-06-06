#pragma once

#define MAX_ELEM 1e9

#include "layer_base.hpp"
#include <stdexcept>
#include <type_traits>
#include "Random.h"

using namespace Eigen;

/*
Fully connected layer X[l]
_batch_size - size of the incoming minibatch (N)
_in_dim - input dimension (D)
-out_dim - output dimension (M)
*/
class Linear : LayerBase {

public:
  Linear(Index _batch_size, Index _in_dim, Index _out_dim, bool _use_bias = false) :
    batch_size(_batch_size),
    in_dim(_in_dim),
    out_dim(_out_dim),
    use_bias(_use_bias) {

    biased_dim = use_bias ? in_dim + 1 : in_dim;
  }

  // prev_layer_out: X[l-1], dim: [N, D]
  virtual void forward(MatrixXd& prev_layer) {

    // dims: [N, D] * [D, M] -> [N, M]

    layer_output.noalias() = prev_layer * weights;

    // we assume that bias adjustment is done at the end of the forward pass
    // since weights will include a bias dimension at initialization, we don't need
    // to worry about biases anymore
    // (X,1) * W dim: [N, D+1] * [D + 1, M]
    if (use_bias) {
      adjust_linear_bias(layer_output);
    }
  }

  // next_layer_grad: delta[l+1] = dL/dX[l+1], dims: [N, M] (same as X[l+1])
  virtual void backward(const MatrixXd& prev_layer, const MatrixXd& next_layer_grad) {
    // this will be fed to the previous backprop layer as the delta parameter
    // dim delta[l+1] * w.T: [N, M] * [M, D] -> [N, D] (same as X[l-1])
    layer_grad_loss_by_input.noalias() = next_layer_grad * weights.transpose();

    //dim X[l].T * delta[l+1]: [D, N] * [N, M] -> [D, M], same as W
    layer_grad_loss_by_weight.noalias() = prev_layer.transpose() * next_layer_grad;
  }

  void init(RNG rng, double mu, double sigma, bool debug=false) {
    if (in_dim <= 0 || out_dim <= 0 || in_dim > MAX_ELEM || out_dim > MAX_ELEM) {
      throw std::invalid_argument("inappropriate dimensions");
    }

    //weights of dimension (D, M)
    weights.resize(biased_dim, out_dim);
    if (debug) {
        init_debug();
        return;
    }
    set_normal_random(weights.data(), weights.size(), rng, mu, sigma);
  }

  // this will be fed to compute dL/dW[l-1]
  // it is dL/dX[l]
  MatrixXd& get_loss_by_input_derivative() {
    return layer_grad_loss_by_input;
  }

  // feed to optimizer
  MatrixXd& get_loss_by_weights_derivative() {
    return layer_grad_loss_by_weight;
  }

  MatrixXd& get_output() {
      return layer_output;
  }

  MatrixXd& get_weights() {
      return weights;
  }

private:

    // init weights to 1 for debugging
    void init_debug() {
        weights = MatrixXd::Ones(biased_dim, out_dim);
    }

  MatrixXd weights;
  MatrixXd layer_output, layer_grad_loss_by_weight, layer_grad_loss_by_input;

  const bool use_bias;
  const Index in_dim, out_dim, batch_size;
  int biased_dim;
};