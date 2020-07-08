#pragma once

#include "opsbase.hpp"

namespace EigenSinn {

  template <typename Scalar, Index Dim>
  inline auto is_elementwise_approx_eq(Tensor<Scalar, Dim> a, Tensor<Scalar, Dim> b, float prec = 1e-5) {

    Tensor<Scalar, Dim> diff = a - b;
    Tensor<Scalar, 0> res = diff.maximum();
    return res(0) <= prec;
  }
}