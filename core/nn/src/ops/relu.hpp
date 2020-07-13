#pragma once

#include "opsbase.hpp"
#include <ops/conversions.hpp>
#include <tuple>

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
  inline auto leaky_relu(const Tensor<Scalar, Rank>& t, float threshold=0.01) {

    auto flat_dim = get_flat_dims(t);

    Tensor<float, 1> mask(flat_dim);
    Tensor<Scalar, Rank> output(t.dimensions());
    output = t;
    mask.setZero();

    array<Index, 1> flat_dim_arr = { flat_dim };
    TensorMap<Tensor<Scalar, 1>> flat(output.data(), flat_dim);

    for (Index i = 0; i < flat_dim; i++) {
      if (flat(i) < 0) {
        flat(i) *= threshold;
        mask(i) = threshold;
      }
      else {
        mask(i) = 1.;
      }
    }

    return Tuple(mask, output);
  }

  template<typename Scalar, Index Rank>
  inline auto leaky_relu_back(const Tensor<Scalar, Rank>& next_layer_grad, const Tensor<float, 1>& mask) {

    Tensor<Scalar, Rank> output;

    output = next_layer_grad * mask;
    return output;
  }
  
}