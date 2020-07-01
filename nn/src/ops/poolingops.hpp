#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <tuple>
#include <stdexcept>
#include "opsbase.hpp"

namespace EigenSinn {

  enum PoolType : int{
    max,
    avg
  };

  template <int Rank, class Dims>
  inline bool check_valid_params(const array<int, Rank/2>& extents, int stride, Dims& dims) {

    if (stride <= 0) {
      return false;
    }

    if ((Rank == 2 || Rank == 4) && Rank / 2 != extents.size()) {

      return false;
    }

    for (int i = 0; i < Rank/2; i++) {
      // we are interested in the second or second and third dimensions
      // depending on the tensor: 2d or 4d
      int tensor_dim = dims[i + 1];
      int diff = tensor_dim - extents[i];

      if (diff < 0) {
        return false;
      }

      if (stride != 1 && diff % stride != 0) {
        return false;
      }
    }

    return true;
  }

  

  template <int Rank, int Dim>
  inline auto do_pool(NnTensor<Rank>& t, const array<int, Dim>& extents, int stride) {
    
    auto dims = t.dimensions();
    
    if (!check_valid_params(extens, stirde, dims)) {

      throw std::invalid_argument("Invalid pooling dimensions");
    }



  }
}