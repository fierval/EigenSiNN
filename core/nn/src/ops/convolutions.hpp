#pragma once

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include <cassert>
#include <iostream>
#include <utility>

#define MAX_PAD 1e6
using namespace Eigen;

namespace EigenSinn {
  
  typedef array<std::pair<int, int>, 4> Padding;
  typedef IndexPair<int> Dim2D;
  typedef IndexPair<Index> Padding2D;

  enum class ConvType : short {
    valid,
    same,
    full
  };

  enum class ConvDims : int {
    batch = 0,
    channel = 1,
    height = 2,
    width = 3
  };

  inline Padding pad2dim(int dim1, int dim2, int dim1_1, int dim2_1) {

    assert(dim1 >= 0 && dim1 < MAX_PAD && dim2 >= 0 && dim2 < MAX_PAD && dim2_1 < MAX_PAD && dim1_1 < MAX_PAD);

    Padding paddings;
    paddings[(int)ConvDims::batch] = std::make_pair(0, 0);
    paddings[(int)ConvDims::height] = std::make_pair(dim1, dim1_1);
    paddings[(int)ConvDims::width] = std::make_pair(dim2, dim2_1);
    paddings[(int)ConvDims::channel] = std::make_pair(0, 0);

    return paddings;
  }

  // output dimensions for a convolution with constant padding
  template <typename Scalar, Index Rank=4>
  inline auto get_output_dimensions(const Tensor<Scalar, Rank>& input, const Tensor<Scalar, Rank>& kernel, const Padding2D& padding, const Index stride = 1) {

    assert(kernel.dimension((int)ConvDims::channel) == input.dimension((int)ConvDims::channel));
    assert(kernel.dimension((int)ConvDims::height) > 0 && kernel.dimension((int)ConvDims::width) > 0);

    Tensor<Scalar, Rank>::Dimensions out_dims;
    out_dims[(int)ConvDims::batch] = input.dimension((int)ConvDims::batch);
    out_dims[(int)ConvDims::height] = (input.dimension((int)ConvDims::height) + 2 * padding.first - kernel.dimension((int)ConvDims::height)) / stride + 1;
    out_dims[(int)ConvDims::width] = (input.dimension((int)ConvDims::width) + 2 * padding.second - kernel.dimension((int)ConvDims::width)) / stride + 1;
    out_dims[(int)ConvDims::channel] = kernel.dimension((int)ConvDims::batch);

    return out_dims;
  }

  template <typename Scalar, Index Rank = 4>
  inline auto get_output_dimensions(const Tensor<Scalar, Rank>& input, const Tensor<Scalar, Rank>& kernel, const ConvType paddingType, const Index stride = 1) {

    switch (paddingType) {
    case ConvType::valid:
      return get_output_dimensions(input, kernel, { 0, 0 }, stride);
    case ConvType::same:
      // in the case of actual convolutions we enforce same padding on "both sides" of a dimension
      assert((kernel.dimension((int)ConvDims::height) & 0x1) != 0 && (kernel.dimension((int)ConvDims::width) & 0x1) != 0);
      return get_output_dimensions(input, kernel, { (kernel.dimension((int)ConvDims::height) - 1) / 2, (kernel.dimension((int)ConvDims::width) - 1) / 2 }, stride);
    case ConvType::full:
      return get_output_dimensions(input, kernel, { kernel.dimension((int)ConvDims::height) - 1, kernel.dimension((int)ConvDims::width) - 1 }, stride);
    default:
      assert(false);
    }
  }

  // NCHW format
  //TODO: support stride
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, Padding padding, Index stride = 1) {

    //dimensions involved in the convolution. Channel dimension is also involved.
    array<Index, 3> dims({ (int)ConvDims::channel, (int)ConvDims::height, (int)ConvDims::width });

    assert(input.dimension((int)ConvDims::channel) == kernel.dimension((int)ConvDims::channel));

    // Pad if apropriate
    Tensor<Scalar, Rank> padded = input.pad(padding);

    // output dimensions
    Padding2D heightWidthPadding = IndexPair(padded.dimension((int)ConvDims::height), padded.dimension((int)ConvDims::width));
    Tensor<Scalar, Rank>::Dimensions out_dims = get_output_dimensions(input, kernel, heightWidthPadding , stride);

    // NCHW output tensor
    Tensor<Scalar, Rank> output(out_dims);

    for (int i = 0; i < kernel.dimension((int)ConvDims::batch); i++) {
      // convolve on 3 dimensions and set the channel dimension of the entire batch
      output.chip(i, (int)ConvDims::channel) = padded.convolve(kernel.chip(i, (int)ConvDims::batch), dims).chip(0, (int)ConvDims::channel);
    }

    return output;
  }

  // pad unevenly in case of k % 2 == 1:
  // more zero's goes upfront
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_same(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel) {
    int dim1 = kernel.dimension((int)ConvDims::height) - 1;
    int dim2 = kernel.dimension((int)ConvDims::width) - 1;
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
    int dim1 = kernel.dimension((int)ConvDims::height) - 1;
    int dim2 = kernel.dimension((int)ConvDims::width) - 1;

    Tensor<Scalar, Rank> output = convolve(input, kernel, pad2dim(dim1, dim2, dim1, dim2));
    return output;
  }
} // namespace EigenSinn
