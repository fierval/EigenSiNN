#pragma once

#include "opsbase.hpp"
#include <ops/conversions.hpp>
#include <tuple>

namespace EigenSinn {

  template<typename Scalar, Index Rank>
  inline auto leaky_relu(Tensor<Scalar, Rank> t, float threshold=0.01) {

    const auto& dims = t.dimensions();
    Index flat_dim = 1;

    for (Index i = 0; i < Rank; i++) {
      flat_dim *= dims[i];
    }


    Tensor<float, 1> mask(flat_dim);
    Tensor<Scalar, Rank> output(dims);
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
  
}