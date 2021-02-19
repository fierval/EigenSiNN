#pragma once

#include "opsbase.hpp"
#include "conversions.hpp"
#include "device/device_tensor.hpp"

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Dim, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const Tensor<Scalar, Dim, Layout>& a, const Tensor<Scalar, Dim, Layout>& b, float prec = 1e-5) {

    Tensor<Scalar, Dim, Layout> diff = a - b;
    Tensor<Scalar, 0, Layout> res = diff.abs().maximum();
    return res(0) <= prec;
  }

  template <typename Scalar, Index Dim, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(Scalar* a, const Tensor<Scalar, Dim, Layout> b, float prec = 1e-5) {

    Tensor<Scalar, Dim, Layout> out = TensorView<Scalar, Dim, Layout>(a, b.dimensions());
    return is_elementwise_approx_eq<Scalar, Dim, Layout>(b, out, prec);
  }

  template <typename Scalar, Index Dim, int Layout = ColMajor>
  inline bool is_elementwise_approx_eq(const Tensor<Scalar, Dim> a, Scalar* b, float prec = 1e-5) {

    Tensor<Scalar, Dim, Layout> out = TensorView<Scalar, Dim, Layout>(b, a.dimensions());
    return is_elementwise_approx_eq<Scalar, Dim, Layout>(a, out, prec);
  }

  template <typename Scalar, Index Dim, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  inline bool is_elementwise_approx_eq(const DeviceTensor<Device_, Scalar, Dim, Layout>& a, std::any b, float prec = 1e-5) {

    DeviceTensor<Device_, Scalar, Dim, Layout> out(b);
    return is_elementwise_approx_eq<Scalar, Dim, Layout, Device_>(a, out, prec);
  }

  template <typename Scalar, Index Dim, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  inline bool is_elementwise_approx_eq(std::any a, const DeviceTensor<Device_, Scalar, Dim, Layout>& b, float prec = 1e-5) {

    DeviceTensor<Device_, Scalar, Dim, Layout> out(a);
    return is_elementwise_approx_eq<Scalar, Dim, Layout, Device_>(b, out, prec);
  }

  template <typename Scalar, Index Dim, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  inline bool is_elementwise_approx_eq(const DeviceTensor<Device_, Scalar, Dim, Layout>& a, const DeviceTensor<Device_, Scalar, Dim, Layout>& b, float prec = 1e-5) {

    Tensor<Scalar, Dim, Layout> ta = a.to_host();
    Tensor<Scalar, Dim, Layout> tb = b.to_host();
    return is_elementwise_approx_eq(ta, tb, prec);
  }

  template <typename Scalar, Index Dim, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  inline bool is_elementwise_approx_eq(const Tensor<Scalar, Dim, Layout>& a, std::any b, float prec = 1e-5) {

    auto out = DeviceTensor<Device_, Scalar, Dim, Layout>(b).to_host();
    return is_elementwise_approx_eq<Scalar, Dim, Layout>(a, out, prec);
  }

  template <typename Scalar, Index Dim, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  inline bool is_elementwise_approx_eq(std::any a, const Tensor<Scalar, Dim, Layout>& b, float prec = 1e-5) {

    auto out = DeviceTensor<Device_, Scalar, Dim, Layout>(a).to_host();
    return is_elementwise_approx_eq<Scalar, Dim, Layout, Device_>(b, out, prec);
  }

}
