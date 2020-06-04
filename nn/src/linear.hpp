#pragma once

#define MAX_ELEM 1e9

#include "layer_base.hpp"
#include <stdexcept>
#include <type_traits>

using namespace Eigen;

/*
Fully connected layer X[l]
_batch_size - size of the incoming minibatch (N)
_in_dim - input dimension (D)
-out_dim - output dimension (M)
*/
class Linear : LayerBase {

public:
  Linear(int _batch_size, int _in_dim, int _out_dim, bool _use_bias = false) :
    batch_size(_batch_size),
    in_dim(_in_dim),
    out_dim(_out_dim),
    use_bias(_use_bias) {


    biased_dim = use_bias ? in_dim + 1 : in_dim;
  }

  // prev_layer_out: X[l-1], dim: [N, D]
  virtual void forward(const MatrixXd& prev_layer_out) {

    MatrixXd& prev_layer = const_cast<MatrixXd&>(prev_layer_out);

    if (use_bias) {
      prev_layer.conservativeResize(prev_layer_out.rows(), prev_layer_out.cols() + 1);
      prev_layer.col(prev_layer.cols() - 1) = VectorXd::Ones(prev_layer.rows());
    }

    // dims: [N, D] * [D, M] -> [N, M]
    layer_output.noalias() = prev_layer * weights.transpose();
  }

  // next_layer_grad: delta[l+1] = dL/dX[l+1], dims: [N, M] (same as X[l+1])
  virtual void backward(const MatrixXd& next_layer_grad) {

    //dim X[l].T * delta[l+1]: [D, N] * [N, M]
    layer_grad_loss_by_weight.noalias() = layer_output.transpose() * next_layer_grad;

    // this will be fed to the previous backprop layer as the delta parameter
    // dim delta[l+1] * w.T: [N, M] * [M, D] -> [N, D] (same as X[l])
    layer_grad_loss_by_output.noalias() = next_layer_grad * weights.transpose();
  }

  void init() {
    if (in_dim <= 0 || out_dim <= 0 || in_dim > MAX_ELEM || out_dim > MAX_ELEM) {
      throw std::invalid_argument("inappropriate dimensions");
    }

    //weights of dimension (D, M)
    weights.resize(biased_dim, out_dim);
  }

  MatrixXd GetGradByX() const {
    return layer_grad_loss_by_output;
  }

  MatrixXd GetWeightsCorrection() const {
    return layer_grad_loss_by_weight;
  }

private:
  MatrixXd weights;
  MatrixXd layer_output, layer_grad_loss_by_weight, layer_grad_loss_by_output;

  const bool use_bias;
  const int in_dim, out_dim, batch_size;
  int biased_dim;
};