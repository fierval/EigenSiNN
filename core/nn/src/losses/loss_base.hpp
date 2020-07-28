#pragma once

#include <any>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank>
  class LossBase {

  public:

    virtual void forward(std::any predictions_any, std::any actual_any) = 0;

    virtual void backward() = 0;

    virtual const std::any get_output() {
      return loss;
    }

    virtual const std::any get_loss_derivative_by_input() {
      return dloss;
    }

  protected:
    array<Index, Rank> orig_dims;
    Scalar loss;
    bool is_dim_set;

    Tensor<Scalar, Rank> dloss;
    Tensor<Scalar, Rank> spread_grad;
  };
}