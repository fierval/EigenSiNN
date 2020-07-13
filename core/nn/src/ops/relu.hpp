#pragma once

#include "opsbase.hpp"
#include <ops/conversions.hpp>
#include <tuple>

namespace EigenSinn {

  template<typename Scalar, Rank>
  inline auto relu(Tensor<Scalar, Rank> t, float threshold = 0f) {

    const auto& dims = t.dimensions;
    int flat_dim = 1;

    for (Index i = 0; i < Rank; i++) {
      flat_dim *= dims[i];
    }


    Tensor<byte, 1> mask(flat_dim);
    Tensor<Scalar, Rank> output(dims) = t;
    mask.setZero();

    Tensor<Scalar, 1> flat = output.reshape(array<int, 1>{flat_dim});

    for (Index i = 0; i < flat_dim; i++) {
      if (output(i) <= threshold) {
        output(i) = 0;
        mask(i) = 1;
      }
    }

    return Tuple(mask, output);
  }
  
}