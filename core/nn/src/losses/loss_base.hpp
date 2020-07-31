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

    inline auto initialize_and_convert(std::any predicted_any, std::any actual_any) {

      Tensor<Scalar, Rank> predicted = from_any<Scalar, Rank>(predicted_any);
      Tensor<Scalar, Rank> actual = from_any<Scalar, Rank>(actual_any);

      array<Index, Rank> predicted_dims = predicted.dimensions();
      array<Index, Rank> actual_dims = actual.dimensions();

      if (!is_dim_set) {
        orig_dims = actual.dimensions();
        spread_grad.resize(orig_dims);
        spread_grad.setConstant(1. / (orig_dims[0] * orig_dims[1]));
        is_dim_set = true;
      }

      for (int i = 0; i < Rank; i++) {
        assert(predicted_dims[i] == orig_dims[i]);
      }

      return Tuple(predicted, actual);
    }

    array<Index, Rank> orig_dims;
    Scalar loss;
    bool is_dim_set;

    Tensor<Scalar, Rank> dloss;
    Tensor<Scalar, Rank> spread_grad;
  };
}