#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include <cassert>
#include <iostream>
#include <utility>
#include <tuple>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <cmath>
#include <numeric>
#include <execution>
#include <vector>
#include <array>
#include <optional>

using namespace Eigen;

#include "device/device_wrapper.hpp"

namespace EigenSinn {

  template<typename Scalar, Index Rank, int Layout = ColMajor>
  using TensorView = TensorMap<Tensor<Scalar, Rank, Layout>>;
  
  template<typename Scalar, Index Rank, int Layout = ColMajor>
  using OptionalTensorView = std::optional<TensorView<Scalar, Rank, Layout>>;

  typedef array<std::pair<Index, Index>, 4> Padding;
  typedef IndexPair<Index> Dim2D;
  typedef IndexPair<long> Padding2D;

  typedef array<IndexPair<int>, 1> ProductDims;

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

  template<int Rank>
  inline array<Index, Rank> reverse_dims(const array<Index, Rank>& dims) {
    array<Index, Rank> rev;

    for (Index i = rev.size() - 1; i > 0; i--) {
      rev[rev.size() - i - 1] = i;
    }
    return rev;
  }

  template <Index Rank>
  inline DSizes<Index, Rank> empty_dims() {
    DSizes<Index, Rank> out;

    for (int i = 0; i < Rank; i++) {
      out[i] = 0;
    }

    return out;
  }

  // convert between dimensions and vector types
  template<Index Rank>
  static inline std::vector<Index> dims2vec(const DSizes<Index, Rank>& dims) {
    std::vector<Index> out(Rank);

    for (Index i = 0; i < Rank; i++) {
      out[i] = dims[i];
    }
    return out;
  }

  template<Index Rank>
  static inline DSizes<Index, Rank> vec2dims(const std::vector<Index>& dims) {
    DSizes<Index, Rank> out;

    for (Index i = 0; i < Rank; i++) {
      out[i] = dims[i];
    }
    return out;
  }


}


