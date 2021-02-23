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
    return is_elementwise_approx_eq<Scalar, Rank, Layout>(b, out, prec);
  }

  template <typename Scalar, Index Rank, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const Tensor<Scalar, Rank> a, Scalar* b, float prec = 1e-5) {

    Tensor<Scalar, Rank, Layout> out = TensorView<Scalar, Rank, Layout>(b, a.dimensions());
    return is_elementwise_approx_eq<Scalar, Rank, Layout>(a, out, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const DeviceTensor<Scalar, Rank, Device_, Layout>& a, std::any b, float prec = 1e-5) {

    DeviceTensor<Scalar, Rank, Device_, Layout> out(b);
    return is_elementwise_approx_eq<Scalar, Rank, Layout, Device_>(a, out, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(std::any a, const DeviceTensor<Scalar, Rank, Device_, Layout>& b, float prec = 1e-5) {

    DeviceTensor<Scalar, Rank, Device_, Layout> out(a);
    return is_elementwise_approx_eq<Scalar, Rank, Layout, Device_>(b, out, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const DeviceTensor<Scalar, Rank, Device_, Layout>& a, const DeviceTensor<Scalar, Rank, Device_, Layout>& b, float prec = 1e-5) {

    Tensor<Scalar, Rank, Layout> ta = a.to_host();
    Tensor<Scalar, Rank, Layout> tb = b.to_host();
    return is_elementwise_approx_eq(ta, tb, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const Tensor<Scalar, Rank, Layout>& a, std::any b, float prec = 1e-5) {

    auto out = DeviceTensor<Scalar, Rank, Device_, Layout>(b).to_host();
    return is_elementwise_approx_eq<Scalar, Rank, Layout>(a, out, prec);
  }

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(std::any a, const Tensor<Scalar, Rank, Layout>& b, float prec = 1e-5) {

    auto out = DeviceTensor<Scalar, Rank, Device_, Layout>(a).to_host();
    return is_elementwise_approx_eq<Scalar, Rank, Layout, Device_>(b, out, prec);
  }

}
