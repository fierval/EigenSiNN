#pragma once

#include "opsbase.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank>
  inline auto get_flat_dims(Tensor<Scalar, Rank> t) {
    const auto& dims = t.dimensions();
    Index flat_dim = 1;

    for (Index i = 0; i < Rank; i++) {
      flat_dim *= dims[i];
    }
    return flat_dim;
  }

  template<typename Scalar, Index Rank>
  inline auto leaky_relu(const Tensor<Scalar, Rank>& t, float threshold) {

    auto flat_dim = get_flat_dims(t);

    Tensor<Scalar, Rank> mask(t.dimensions());
    Tensor<Scalar, Rank> output(t.dimensions());
    output = t;
    mask.setZero();

    array<Index, 1> flat_dim_arr = { flat_dim };
    TensorMap<Tensor<Scalar, 1>> flat_output(output.data(), flat_dim);
    TensorMap<Tensor<Scalar, 1>> flat_mask(mask.data(), flat_dim);

    std::vector<Index> range(flat_dim);
    std::iota(range.begin(), range.end(), 0);

    std::for_each(std::execution::par, range.begin(), range.end(), [&](Index i) {
      if (flat_output(i) < 0) {
        flat_output(i) *= threshold;
        flat_mask(i) = threshold;
      }
      else {
        flat_mask(i) = 1.;
      }
      });

    return Tuple(mask, output);
  }

  template<typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  inline auto leaky_relu_back(const Tensor<Scalar, Rank>& next_layer_grad, const Tensor<Scalar, Rank>& mask, const Device_& device = DefaultDevice()) {

    Tensor<Scalar, Rank> output(next_layer_grad.dimensions());

    output.device(device) = next_layer_grad * mask;
    return output;
  }

}