#pragma once

#include <any>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank>
  class LossBase {

  public:
    virtual void init(const array<Index, Rank>& _orig_dims) {
      orig_dims = _orig_dims;
    }
    virtual void compute(std::any predictions_any, std::any actual_any) = 0;

    virtual void backward() = 0;

    virtual const std::any get_output() {
      return loss;
    }

  protected:
    array<Index, Rank> orig_dims;
    Scalar loss;
  };
}