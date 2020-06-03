#pragma once

#define MAX_ELEM 1e9

#include "layer_base.hpp"
#include <stdexcept>
#include <type_traits>

using namespace Eigen;

class Linear : LayerBase
{

public:
  Linear(int _in_dim, int _out_dim, bool _use_bias = false)
      : use_bias(_use_bias),
        in_dim(_in_dim),
        out_dim(_out_dim)
  {

    biased_dim = use_bias ? in_dim + 1 : in_dim;
  }

  virtual void forward(const MatrixXd &prev_layer_out)
  {

    MatrixXd &prev_layer = const_cast<MatrixXd &>(prev_layer_out);

    if (use_bias) {
      prev_layer.conservativeResize(prev_layer_out.rows(), prev_layer_out.cols() + 1);
      prev_layer.col(prev_layer.cols() - 1) = VectorXd::Constant(prev_layer.rows(), 1);
    }

    layer_output.noalias() = weights * prev_layer;
  }

  virtual void backward(const MatrixXd &prev_layer_out, const MatrixXd &next_layer_grad)
  {
  }

  void init()
  {
    if (in_dim <= 0 || out_dim <= 0 || in_dim > MAX_ELEM || out_dim > MAX_ELEM)
    {
      throw std::invalid_argument("inappropriate dimensions");
    }

    weights.resize(out_dim, biased_dim);
  }

private:
  MatrixXd weights;
  VectorXd layer_output;

  const bool use_bias;
  const int in_dim, out_dim;
  int biased_dim;
};