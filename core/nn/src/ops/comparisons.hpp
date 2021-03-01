#pragma once

#include "opsbase.hpp"
#include "conversions.hpp"
#include "device/device_tensor.hpp"

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const Tensor<Scalar, Rank, Layout>& a, const Tensor<Scalar, Rank, Layout>& b, float prec = 1e-5) {

    Tensor<Scalar, Rank, Layout> diff = a - b;
    Tensor<Scalar, 0, Layout> res = diff.abs().maximum();
    return res(0) <= prec;
  }

  template <typename Scalar, Index Rank, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(Scalar* a, const Tensor<Scalar, Rank, Layout> b, float prec = 1e-5) {

    Tensor<Scalar, Rank, Layout> out = TensorView<Scalar, Rank, Layout>(a, b.dimensions());
    return is_elementwise_approx_eq(b, out, prec);
  }

  template <typename Scalar, Index Rank, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const Tensor<Scalar, Rank> a, Scalar* b, float prec = 1e-5) {

    Tensor<Scalar, Rank, Layout> out = TensorView<Scalar, Rank, Layout>(b, a.dimensions());
    return is_elementwise_approx_eq(a, out, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const DeviceTensor<Scalar, Rank, Device_, Layout>& a, const DeviceTensor<Scalar, Rank, Device_, Layout>& b, float prec = 1e-5) {

    Tensor<Scalar, Rank, Layout> ta = a.to_host();
    Tensor<Scalar, Rank, Layout> tb = b.to_host();
    return is_elementwise_approx_eq(ta, tb, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const DeviceTensor<Scalar, Rank, Device_, Layout>& a, const Tensor<Scalar, Rank, Layout>& b, float prec = 1e-5) {

    auto out = a.to_host();
    return is_elementwise_approx_eq(out, b, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const Tensor<Scalar, Rank,Layout>& a, const DeviceTensor<Scalar, Rank, Device_, Layout>& b, float prec = 1e-5) {

    const auto out = b.to_host();
    return is_elementwise_approx_eq(a, out, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const DeviceTensor<Scalar, Rank, Device_, Layout>& a, PtrTensorAdapter<Scalar, Device_>& b, float prec = 1e-5) {

    auto out = a.to_host();
    DeviceTensor<Scalar, Rank, Device_, Layout> tmp(b);
    return is_elementwise_approx_eq(out, tmp, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const PtrTensorAdapter<Scalar, Device_>& a, const DeviceTensor<Scalar, Rank, Device_, Layout>& b, float prec = 1e-5) {

    const auto out = b.to_host();
    DeviceTensor<Scalar, Rank, Device_, Layout> tmp(a);
    return is_elementwise_approx_eq(tmp, out, prec);
  }

}
