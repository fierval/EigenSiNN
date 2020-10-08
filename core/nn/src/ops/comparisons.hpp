#pragma once

#include "opsbase.hpp"
#include "conversions.hpp"

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Dim>
  inline auto is_elementwise_approx_eq(const Tensor<Scalar, Dim>& a, const Tensor<Scalar, Dim>& b, float prec = 1e-5) {

    Tensor<Scalar, Dim> diff = a - b;
    Tensor<Scalar, 0> res = diff.abs().maximum();
    return res(0) <= prec;
  }

  template <typename Scalar, Index Dim>
  inline auto is_elementwise_approx_eq(Scalar * a, const Tensor<Scalar, Dim> b, float prec = 1e-5) {

    TensorMap<Tensor<Scalar, Dim>> out(a, b.dimensions());
    return is_elementwise_approx_eq<Scalar, Dim>(out, b, prec);
  }

  template <typename Scalar, Index Dim>
  inline auto is_elementwise_approx_eq(const Tensor<Scalar, Dim> a, Scalar * b, float prec = 1e-5) {

    TensorMap<Tensor<Scalar, Dim>> out(b, a.dimensions());
    return is_elementwise_approx_eq<Scalar, Dim>(a, out, prec);
  }
}
