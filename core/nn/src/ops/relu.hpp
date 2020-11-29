#pragma once

#include "opsbase.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank, int Layout, typename Device_>
  inline auto get_flat_dims(const DeviceTensor<Device_, Scalar, Rank, Layout>& t) {
    const auto& dims = t.dimensions();
    Index flat_dim = 1;

    for (Index i = 0; i < Rank; i++) {
      flat_dim *= dims[i];
    }
    return flat_dim;
  }

  template<typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto leaky_relu(DeviceTensor<Device_, Scalar, Rank, Layout>& t, Scalar threshold) {

    auto flat_dim = get_flat_dims(t);
    
    DeviceTensor<Device_, Scalar, Rank, Layout> pos_mask(t.dimensions());
    DeviceTensor<Device_, Scalar, Rank, Layout> neg_mask(t.dimensions());
    DeviceTensor<Device_, Scalar, Rank, Layout> mask(t.dimensions());
    DeviceTensor<Device_, Scalar, Rank, Layout> output(t.dimensions());
    
    output = t;
    pos_mask.view() = (*output >= static_cast<Scalar>(0)).cast<Scalar>();
    neg_mask.view() = (*output < static_cast<Scalar>(0)).cast<Scalar>();
    
    neg_mask *= threshold;
    mask = neg_mask + pos_mask;

    output = mask * t;
    return Tuple(mask, output);
  }

  template<typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto leaky_relu_back(DeviceTensor<Device_, Scalar, Rank, Layout>& next_layer_grad, DeviceTensor<Device_, Scalar, Rank, Layout>& mask) {

    DeviceTensor<Device_, Scalar, Rank, Layout> output(next_layer_grad.dimensions());

    output = next_layer_grad * mask;
    return output;
  }

}