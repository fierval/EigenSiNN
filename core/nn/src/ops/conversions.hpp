#pragma once
#include "opsbase.hpp"

using namespace Eigen;

namespace EigenSinn {

  template<typename Scalar>
  inline auto from_binary_to_category(const Tensor<Scalar, 2>& inp) {

    Tensor<Scalar, 1> cat(inp.dimension(0));

    for (Index row = 0; row < inp.dimension(0); row++) {
      for (Index col = 0; col < inp.dimension(1); col++) {
        if (inp(row, col) > 0) {
          cat(row) = col;
          break;
        }
        if (col == inp.dimension(1)) {
          throw std::logic_error("row without non-zero element");
        }
      }
    }

    return cat;
  }
}