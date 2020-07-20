#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/Tensor"
#include <cassert>
#include <iostream>
#include <utility>
#include <any>
#include <tuple>
#include <stdexcept>
#include <cstring>

using namespace Eigen;

namespace EigenSinn {
  template <int Rank>
  using NnTensor = Tensor<float, Rank>;

  typedef array<std::pair<Index, Index>, 4> Padding;
  typedef IndexPair<Index> Dim2D;
  typedef IndexPair<Index> Padding2D;

  enum class ConvType : short {
    valid,
    same,
    full
  };

  enum class ImageDims : int {
    batch = 0,
    channel = 1,
    height = 2,
    width = 3
  };

  enum PoolType : int {
    max,
    avg
  };

}


