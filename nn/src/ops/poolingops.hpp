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

  template <int Rank, int Dim, class Dims>
  inline bool check_valid_params(const array<int, Dim>& extents, int stride, Dims& dims) {

    bool res = false;
    
    return res;
  }

  template <int Rank, int Dim>
  inline auto do_pool(NnTensor<Rank>& t, const array<int, Dim>& extents, int stride) {
    
    NnTensor<Rank>::Dimensions dims = t.dimensions();
    
    if (!check_valid_params(extens, stirde, dims)) {

      throw std::invalid_argument("Invalid pooling dimensions");
    }



  }
}