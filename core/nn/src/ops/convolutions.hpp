#pragma once

#include "opsbase.hpp"

#define MAX_PAD 1e6
using namespace Eigen;

namespace EigenSinn {
  
  inline Padding pad2dim(int dim1, int dim2, int dim1_1, int dim2_1) {

    assert(dim1 >= 0 && dim1 < MAX_PAD && dim2 >= 0 && dim2 < MAX_PAD && dim2_1 < MAX_PAD && dim1_1 < MAX_PAD);

    Padding paddings;
    paddings[(int)ImageDims::batch] = std::make_pair(0, 0);
    paddings[(int)ImageDims::height] = std::make_pair(dim1, dim1_1);
    paddings[(int)ImageDims::width] = std::make_pair(dim2, dim2_1);
    paddings[(int)ImageDims::channel] = std::make_pair(0, 0);

    return paddings;
  }

  // output dimensions for a convolution with constant padding
  template <typename Scalar, Index Rank=4>
  inline auto get_output_dimensions(const Tensor<Scalar, Rank>& input, const Tensor<Scalar, Rank>& kernel, const Padding2D& padding, const Index stride = 1) {

    assert(kernel.dimension((int)ImageDims::channel) == input.dimension((int)ImageDims::channel));
    assert(kernel.dimension((int)ImageDims::height) > 0 && kernel.dimension((int)ImageDims::width) > 0);

    Tensor<Scalar, Rank>::Dimensions out_dims;
    out_dims[(int)ImageDims::batch] = input.dimension((int)ImageDims::batch);
    out_dims[(int)ImageDims::height] = (input.dimension((int)ImageDims::height) + 2 * padding.first - kernel.dimension((int)ImageDims::height)) / stride + 1;
    out_dims[(int)ImageDims::width] = (input.dimension((int)ImageDims::width) + 2 * padding.second - kernel.dimension((int)ImageDims::width)) / stride + 1;
    out_dims[(int)ImageDims::channel] = kernel.dimension((int)ImageDims::batch);

    return out_dims;
  }

  template <typename Scalar, Index Rank = 4>
  inline auto get_output_dimensions(const Tensor<Scalar, Rank>& input, const Tensor<Scalar, Rank>& kernel, const ConvType paddingType, const Index stride = 1) {

    switch (paddingType) {
    case ConvType::valid:
      return get_output_dimensions(input, kernel, { 0, 0 }, stride);
    case ConvType::same:
      // in the case of actual convolutions we enforce same padding on "both sides" of a dimension
      assert((kernel.dimension((int)ImageDims::height) & 0x1) != 0 && (kernel.dimension((int)ImageDims::width) & 0x1) != 0);
      return get_output_dimensions(input, kernel, { (kernel.dimension((int)ImageDims::height) - 1) / 2, (kernel.dimension((int)ImageDims::width) - 1) / 2 }, stride);
    case ConvType::full:
      return get_output_dimensions(input, kernel, { kernel.dimension((int)ImageDims::height) - 1, kernel.dimension((int)ImageDims::width) - 1 }, stride);
    default:
      assert(false);
    }
  }

  // NCHW format
  //TODO: support stride
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, Padding padding, Index stride = 1) {

    //dimensions involved in the convolution. Channel dimension is also involved.
    array<Index, 3> dims({ (int)ImageDims::channel, (int)ImageDims::height, (int)ImageDims::width });

    assert(input.dimension((int)ImageDims::channel) == kernel.dimension((int)ImageDims::channel));

    // Pad if apropriate
    Tensor<Scalar, Rank> padded = input.pad(padding);

    // output dimensions
    Padding2D heightWidthPadding = IndexPair(padded.dimension((int)ImageDims::height), padded.dimension((int)ImageDims::width));
    Tensor<Scalar, Rank>::Dimensions out_dims = get_output_dimensions(input, kernel, heightWidthPadding , stride);

    // NCHW output tensor
    Tensor<Scalar, Rank> output(out_dims);

    for (int i = 0; i < kernel.dimension((int)ImageDims::batch); i++) {
      // convolve on 3 dimensions and set the channel dimension of the entire batch
      output.chip(i, (int)ImageDims::channel) = padded.convolve(kernel.chip(i, (int)ImageDims::batch), dims).chip(0, (int)ImageDims::channel);
    }

    return output;
  }

  // pad unevenly in case of k % 2 == 1:
  // more zero's goes upfront
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_same(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel) {
    int dim1 = kernel.dimension((int)ImageDims::height) - 1;
    int dim2 = kernel.dimension((int)ImageDims::width) - 1;
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
    int dim1 = kernel.dimension((int)ImageDims::height) - 1;
    int dim2 = kernel.dimension((int)ImageDims::width) - 1;

    Tensor<Scalar, Rank> output = convolve(input, kernel, pad2dim(dim1, dim2, dim1, dim2));
    return output;
  }
} // namespace EigenSinn
