#pragma once

#include <any>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, typename Actual, Index Rank>
  class LossBase {

  public:

    virtual void step(const Tensor<Scalar, Rank>& predictions_any, const Tensor<Actual, Rank>& actual_any) = 0;

    virtual Scalar get_output() {
      return loss;
    }

    virtual Scalar * get_loss_derivative_by_input() {
      return dloss.data();
    }

    const array<Index, Rank>& get_dims() { return orig_dims; }

  protected:

    inline void initialize(const Tensor<Scalar, Rank>& predicted, const Tensor<Actual, Rank> actual) {

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
    }

    array<Index, Rank> orig_dims;
    Scalar loss;
    bool is_dim_set;

    Tensor<Scalar, Rank> dloss;
    Tensor<Scalar, Rank> spread_grad;
  };
}