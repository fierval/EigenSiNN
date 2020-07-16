#pragma once

#include "opsbase.hpp"
#include "conversions.hpp"

namespace EigenSinn {

  template <typename Scalar, Index Dim>
  inline auto is_elementwise_approx_eq(Tensor<Scalar, Dim> a, Tensor<Scalar, Dim> b, float prec = 1e-5) {

    Tensor<Scalar, Dim> diff = a - b;
    Tensor<Scalar, 0> res = diff.abs().maximum();
    return res(0) <= prec;
  }

  template <typename Scalar, Index Dim>
  inline auto is_elementwise_approx_eq(std::any a, Tensor<Scalar, Dim> b, float prec = 1e-5) {

    return is_elementwise_approx_eq(from_any<Scalar, Dim>(a), b, prec);
  }

  template <typename Scalar, Index Dim>
  inline auto is_elementwise_approx_eq(Tensor<Scalar, Dim> a, std::any b, float prec = 1e-5) {

    return is_elementwise_approx_eq(a, from_any<Scalar, Dim>(b), prec);
  }
}
