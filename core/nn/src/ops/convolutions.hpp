#pragma once

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include <cassert>
#include <iostream>
#include <utility>

#define MAX_PAD 1e6
using namespace Eigen;

namespace EigenSinn {
  
  typedef Eigen::array<std::pair<int, int>, 4> Padding;
  typedef Eigen::IndexPair<int> Dim2D;
  typedef Eigen::IndexPair<Eigen::Index> Padding2D;

  enum class ConvType : short {
    valid,
    same,
    full
  };

  inline Padding pad2dim(int dim1, int dim2, int dim1_1, int dim2_1) {

    assert(dim1 >= 0 && dim1 < MAX_PAD && dim2 >= 0 && dim2 < MAX_PAD && dim2_1 < MAX_PAD && dim1_1 < MAX_PAD);

    Padding paddings;
    paddings[0] = std::make_pair(0, 0);
    paddings[1] = std::make_pair(dim1, dim1_1);
    paddings[2] = std::make_pair(dim2, dim2_1);
    paddings[3] = std::make_pair(0, 0);

    return paddings;
  }

  // output dimensions for a convolution with constant padding
  template <typename Scalar, Index Rank=4>
  inline auto get_output_dimensions(const Tensor<Scalar, Rank>& input, const Tensor<Scalar, Rank>& kernel, const Padding2D& padding, const Eigen::Index stride = 1) {

    assert(kernel.dimension(3) == input.dimension(3));
    assert(kernel.dimension(1) > 0 && kernel.dimension(2) > 0);

    Tensor<Scalar, Rank>::Dimensions out_dims;
    out_dims[0] = input.dimension(0);
    out_dims[1] = (input.dimension(1) + 2 * padding.first - kernel.dimension(1)) / stride + 1;
    out_dims[2] = (input.dimension(2) + 2 * padding.second - kernel.dimension(2)) / stride + 1;
    out_dims[3] = kernel.dimension(0);

    return out_dims;
  }

  template <typename Scalar, Index Rank = 4>
  inline auto get_output_dimensions(const Tensor<Scalar, Rank>& input, const Tensor<Scalar, Rank>& kernel, const ConvType paddingType, const Eigen::Index stride = 1) {

    switch (paddingType) {
    case ConvType::valid:
      return get_output_dimensions(input, kernel, { 0, 0 }, stride);
    case ConvType::same:
      // in the case of actual convolutions we enforce same padding on "both sides" of a dimension
      assert((kernel.dimension(1) & 0x1) != 0 && (kernel.dimension(2) & 0x1) != 0);
      return get_output_dimensions(input, kernel, { (kernel.dimension(1) - 1) / 2, (kernel.dimension(2) - 1) / 2 }, stride);
    case ConvType::full:
      return get_output_dimensions(input, kernel, { kernel.dimension(1) - 1, kernel.dimension(2) - 1 }, stride);
    default:
      assert(false);
    }
  }

  // NWHC format
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, Padding padding) {
    Eigen::array<Eigen::Index, 3> dims({ 1, 2, 3 });

    assert(input.dimension(3) == kernel.dimension(3));

    // Pad if apropriate
    Tensor<Scalar, Rank> padded = input.pad(padding).eval();

    // NWHC output tensor
    Tensor<Scalar, Rank> output(padded.dimension(0), padded.dimension(1) - kernel.dimension(1) + 1, padded.dimension(2) - kernel.dimension(2) + 1, kernel.dimension(0));

    for (int i = 0; i < kernel.dimension(0); i++) {
      // final chip(0, 0) removes the third dimension
      output.chip(i, 3) = padded.convolve(kernel.chip(i, 0), dims).chip(0, 3);
    }

    return output;
  }

  // pad unevenly in case of k % 2 == 1:
  // more zero's goes upfront
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_same(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel) {
    int dim1 = kernel.dimension(1) - 1;
    int dim2 = kernel.dimension(2) - 1;
    int dim1_1 = dim1 / 2;
    int dim2_1 = dim2 / 2;
    
    dim1 = (dim1 & 0x1) == 0 ? dim1_1 : dim1_1 + 1;
    dim2 = (dim2 & 0x1) == 0 ? dim2_1 : dim2_1 + 1;

    Tensor<Scalar, Rank> output = convolve(input, kernel, pad2dim(dim1, dim2, dim1_1, dim2_1));
    return output;
  }

  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_valid(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel) {
    Tensor<Scalar, Rank> output = convolve(input, kernel, pad2dim(0, 0, 0, 0));
    return output;
  }

  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_full(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel) {
    int dim1 = kernel.dimension(1) - 1;
    int dim2 = kernel.dimension(2) - 1;

    Tensor<Scalar, Rank> output = convolve(input, kernel, pad2dim(dim1, dim2, dim1, dim2));
    return output;
  }
} // namespace EigenSinn
