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

  inline Padding pad2dim(const Padding2D& pad2d) {
    return pad2dim(pad2d.first, pad2d.second, pad2d.first, pad2d.second);
  }

  template <typename Scalar, Index Rank=4>
  inline auto get_output_dimensions(const Tensor<Scalar, Rank>& input, const array<Index, Rank> kernel_dims, const Padding2D& padding, const Index stride = 1) {

    assert(kernel_dims[(int)ImageDims::channel] == input.dimension((int)ImageDims::channel));
    assert(kernel_dims[(int)ImageDims::height] > 0 && kernel_dims[(int)ImageDims::width] > 0);

    Tensor<Scalar, Rank>::Dimensions out_dims;

    // compute total padding. Normally 2xp, but we have "left" and "right" padding amount that may differ
    Index pad_height = padding.first;
    Index pad_width = padding.second;

    assert((input.dimension((int)ImageDims::height) + 2 * pad_height - kernel_dims[(int)ImageDims::height]) % stride == 0);
    assert((input.dimension((int)ImageDims::width) + 2 * pad_width - kernel_dims[(int)ImageDims::width]) % stride == 0);

    out_dims[(int)ImageDims::batch] = input.dimension((int)ImageDims::batch);
    out_dims[(int)ImageDims::height] = (input.dimension((int)ImageDims::height) + pad_height - kernel_dims[(int)ImageDims::height]) / stride + 1;
    out_dims[(int)ImageDims::width] = (input.dimension((int)ImageDims::width) + pad_width - kernel_dims[(int)ImageDims::width]) / stride + 1;
    out_dims[(int)ImageDims::channel] = kernel_dims[(int)ImageDims::batch];

    return out_dims;


  }

  // output dimensions for a convolution with constant padding
  template <typename Scalar, Index Rank=4>
  inline auto get_output_dimensions(const Tensor<Scalar, Rank>& input, const Tensor<Scalar, Rank>& kernel, const Padding2D& padding, const Index stride = 1) {

    return get_output_dimensions(input, kernel.dimensions(), padding, stride);
  }

  // NCHW format
  //TODO: support stride
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, const Padding2D& padding, Index stride = 1) {

    //dimensions involved in the convolution. Channel dimension is also involved.
    array<Index, 3> dims({ (int)ImageDims::channel, (int)ImageDims::height, (int)ImageDims::width });

    assert(input.dimension((int)ImageDims::channel) == kernel.dimension((int)ImageDims::channel));

    // Pad if apropriate
    Tensor<Scalar, Rank> padded = input.pad(pad2dim(padding));

    // output dimensions
    Tensor<Scalar, Rank>::Dimensions out_dims = get_output_dimensions(input, kernel, padding, stride);

    // NCHW output tensor
    Tensor<Scalar, Rank> output(out_dims);

    for (int i = 0; i < kernel.dimension((int)ImageDims::batch); i++) {
      // convolve on 3 dimensions and set the channel dimension of the entire batch
      output.chip(i, (int)ImageDims::channel) = padded.convolve(kernel.chip(i, (int)ImageDims::batch), dims).chip(0, (int)ImageDims::channel);
    }

    return output;
  }

  // NCHW format, col-major storage order
  template <typename Scalar, Index Rank = 4>
  inline auto im2col(const Tensor<Scalar, Rank>& input, const Tensor<Scalar, Rank>& kernel, const Padding2D& padding, Index stride = 1) {

    auto out_dims = get_output_dimensions(input, kernel, padding, stride);
    
    auto col_dim = kernel.dimension(1) * kernel.dimension(2) * kernel.dimension(3);
    array<Index, Rank> starts = { 0, 0, 0, 0 };
    array<Index, Rank> offsets = { input.dimension(0), kernel.dimension(1), kernel.dimension(2), kernel.dimension(3) };

    Tensor<Scalar, 2> output(col_dim, input.dimension(0));
    
    // pad the tensor before we convolve
    Tensor<Scalar, 4> padded = input.pad(pad2dim(padding));

    // we want to take advantage of flattening but
    // our tensors are stored in row-major order.
    // we need to flatten as if it's col-major
    array<int, 4> shuffle_dims = { 3, 2, 1, 0 };

    // "move" the kernel along the batch and convert
    Index col, row;
    for (col = 0, starts[2] = 0; col < out_dims[2]; col++, starts[2] +=stride) {
      for (row = 0, starts[3] = 0; row < out_dims[3]; row++, starts[3] += stride) {

        Tensor<Scalar, Rank> cur_slice = padded.slice(starts, offsets).shuffle(shuffle_dims);
        TensorMap<Tensor<Scalar, 2>> flat_slice(cur_slice.data(), col_dim, input.dimension(0));

        if (col == 0 && row == 0) {
          output = flat_slice;
          continue;
        }

        Tensor<Scalar, 2> tmp = output;
        tmp = output.concatenate(flat_slice, 1);
        output = tmp;
      }

    }
    return output;
  }

  // return kernel representation for GEMM with 
  // im2col representation of the conv layer
  template <typename Scalar>
  inline auto unfold_kernel(const Tensor<Scalar, 4> kernel) {

    auto dims = kernel.dimensions();
    Index col_channels = dims[1] * dims[2] * dims[3];

    Tensor<Scalar, 4> shuffled_kernel = kernel.shuffle({ 0, 3, 2, 1 });
    TensorMap<Tensor<Scalar, 2>> flat_kernel(kernel.data(), dims[0], col_channels);
    return flat_kernel;
  }

  template <typename Scalar>
  inline auto col2im(const Tensor<Scalar, 2>& col, const array<Index, 4>& kernel_dims, const array<Index, 4> orig_dims,  const Padding2D& padding, int stride = 1) {

    Index channels = kernel_dims[1], 
      width = orig_dims[3] + 2 * padding.first, 
      height = orig_dims[2] + 2 * padding.second, 
      batch_size = orig_dims[0];

    array<Index, 2> col_dims = col.dimensions();
    Index total_bytes = batch_size * channels * kernel_dims[1] * kernel_dims[2] * sizeof(Scalar);

    Tensor<Scalar, 4> out(width, height, channels, batch_size);
    
    Index out_w = 0, out_h = 0;
    array<Index, 2> slice_starts = { 0, 0 };
    array<Index, 2> slice_offsets = { col.dimension(0), batch_size };

    // loop over col's batch size at a time
      // figure where it goes into the output
      // memcpy with setValues
    // shuffle dims to batch_size, channels, height, width
    // unpad with slice
    for (Index i = 0; i < col_dims[1] / batch_size; i++, slice_starts[1] += batch_size) {
      
      Tensor<Scalar, 1> slice = col.chip(i, 1);
      Index cur_index = out_w + width * out_h;
      
      memcpy(out.data() + cur_index, slice.data(), total_bytes);
      
      // move to the next slice
      out_w += stride;
      if (out_w + kernel_dims[3] >= width) {
        out_w = 0;
        out_h += stride;
      }
    }
    
    Tensor<Scalar, 4> output(batch_size, channels,  orig_dims[2], orig_dims[3]);
    array<Index, 4> final_starts = { padding.second, padding.first, 0, 0 };
    array<Index, 4> final_lengths = { orig_dims[3], orig_dims[2], channels, batch_size };
    
    //unpad and reshuffle
    output = out.slice(final_starts, final_lengths).shuffle(array<Index, 4>{ 3, 2, 1, 0 });
    return output;
  }

  // pad unevenly in case of k % 2 == 1:
  // more zero's goes upfront
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_same(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel) {
    int dim1 = kernel.dimension((int)ImageDims::height) - 1;
    int dim2 = kernel.dimension((int)ImageDims::width) - 1;
    
    assert(dim1 & 0x1 == 0);
    assert(dim2 & 0x1 == 0);

    Tensor<Scalar, Rank> output = convolve(input, kernel, {dim1 / 2, dim2 / 2});
    return output;
  }

  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_valid(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel) {
    Tensor<Scalar, Rank> output = convolve(input, kernel, { 0, 0 });
    return output;
  }

  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_full(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel) {
    int dim1 = kernel.dimension((int)ImageDims::height) - 1;
    int dim2 = kernel.dimension((int)ImageDims::width) - 1;

    Tensor<Scalar, Rank> output = convolve(input, kernel, { dim1, dim2 });
    return output;
  }
} // namespace EigenSinn
