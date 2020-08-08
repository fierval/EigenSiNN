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

    Index pad_height = 2 * padding.first;
    Index pad_width = 2 * padding.second;

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
  inline auto im2col(const Tensor<Scalar, Rank>& input, const DSizes<Index, 4>& kernel_dims, const Padding2D& padding, Index stride = 1) {

    auto out_dims = get_output_dimensions(input, kernel_dims, padding, stride);
    
    auto col_dim = kernel_dims[1] * kernel_dims[2] * kernel_dims[3];
    array<Index, Rank> starts = { 0, 0, 0, 0 };
    array<Index, Rank> offsets = { input.dimension(0), kernel_dims[1], kernel_dims[2], kernel_dims[3] };

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
        TensorMap<Tensor<Scalar, 2>> flat_slice(cur_slice.data(), col_dim, padded.dimension(0));

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
  inline auto unfold_kernel(Tensor<Scalar, 4> kernel) {

    auto dims = kernel.dimensions();
    Index col_channels = dims[1] * dims[2] * dims[3];

    Tensor<Scalar, 2> flat_kernel = kernel.shuffle(array<Index, 4>{ 0, 3, 2, 1 }).reshape(array<Index, 2>{dims[0], col_channels});
    return flat_kernel;
  }

  // convert back to the [b, c, h, w] kernel representation
  template <typename Scalar>
  inline auto fold_kernel(const Tensor<Scalar, 2>& kernel_col, const array<Index, 4>& expected_dims) {

    assert(expected_dims[0] == kernel_col.dimension(0));
    assert(expected_dims[1] * expected_dims[2] * expected_dims[3] == kernel_col.dimension(1));

    Tensor<Scalar, 4> out = 
      kernel_col.reshape(array<Index, 4>{ expected_dims[0], expected_dims[3], expected_dims[2], expected_dims[1] })
      .shuffle(array<Index, 4>{0, 3, 2, 1});

    return out;
  }

  // return output layer representation for GEMM with 
  // im2col representation of the conv layer
  // final dimensions should be the same as those of FX in col form:
  // F: [Bf x C * Hf * Wf], X: [C * Hf * Wf x B * Ho * Wo] -> [Bf X B * Ho * Wo], 
  // Resulting unrolled dimensions: [B, Bf, Ho, Wo], Bf = new C
  template <typename Scalar>
  inline auto unfold_conv_res(Tensor<Scalar, 4> layer) {

    auto dims = layer.dimensions();
    Index col_channels = dims[0] * dims[2] * dims[3]; // B * Ho * Wo

    Tensor<Scalar, 2> flat_layer = layer.shuffle(array<Index, 4>{1, 0, 3, 2}).reshape(array<Index, 2>{dims[1], col_channels});
    return flat_layer;
  }


  // returns the convolution result in its "normal" form
  // assuming we have just performed a convolution operation.
  // e.g. t (*) k = r, t: [2, 3, 4, 4], k: [5, 3, 3, 3], r: [2, 5, 2, 2]
  // r in im2col form will be [5, 8]
  template <typename Scalar>
  inline auto fold_conv_res(const Tensor<Scalar, 2>& conv_res, const array<Index, 4>& expected_dims) {

    assert(expected_dims[1] == conv_res.dimension(0));
    assert(expected_dims[0] * expected_dims[2] * expected_dims[3] == conv_res.dimension(1));

    Tensor<Scalar, 4> out = conv_res.reshape(array<Index, 4>{ expected_dims[1], expected_dims[0], expected_dims[3], expected_dims[2] }).shuffle(array<Index, 4>{1, 0, 3, 2});
    return out;
  }

  // NOT the reverse of im2col.
  // Used to fold the backward pass result of dX/dL. 
  // We still slide the kernel window over the 2d representation, 
  // Adding contributions of each folded slice to the result
  template <typename Scalar>
  inline auto col2im(const Tensor<Scalar, 2>& col, const array<Index, 4>& kernel_dims, const array<Index, 4> orig_dims,  const Padding2D& padding, int stride = 1) {

    // intermediate output: original dimensions padded
    Index channels = kernel_dims[1], 
      height = orig_dims[2] + 2 * padding.first, 
      width = orig_dims[3] + 2 * padding.second, 
      batch_size = orig_dims[0];

    array<Index, 2> col_dims = col.dimensions();
    Tensor<Scalar, 4> out(batch_size, channels, height, width);
    out.setZero();

    Index out_w = 0, out_h = 0;
    array<Index, 2> slice_starts = { 0, 0 };
    array<Index, 2> slice_offsets = { col.dimension(0), batch_size };
    array<Index, 4> rev_shape = { kernel_dims[3], kernel_dims[2], kernel_dims[1], batch_size};
    
    // loop over col's batch size at a time
      // figure where it goes into the output
      // memcpy with setValues
    // shuffle dims to batch_size, channels, height, width
    // unpad with slice
    for (Index i = 0; i < col_dims[1] / batch_size; i++, slice_starts[1] += batch_size) {
      
      Tensor<Scalar, 4> slice = col.slice(slice_starts, slice_offsets).reshape(rev_shape).shuffle(array<Index, 4>{3, 2, 1, 0});
     
      for (Index b = 0; b < batch_size; b++) {
        for (Index c = 0; c < channels; c++) {
          for (Index h = 0; h < kernel_dims[2]; h++) {
            for (Index w = 0; w < kernel_dims[3]; w++) {
              out(b, c, h + out_h, w + out_w) += slice(b, c, h, w);
            }
          }
        }
      }
      // move to the next slice
      out_w += stride;
      if (out_w + kernel_dims[3] > width) {
        out_w = 0;
        out_h += stride;
      }
    }
    
    //unpad
    array<Index, 4> unpad_starts = { 0, 0, padding.first, padding.second };
    array<Index, 4> unpad_offsets = { out.dimension(0), out.dimension(1), orig_dims[2], orig_dims[3] };
    Tensor<Scalar, 4> output = out.slice(unpad_starts, unpad_offsets);
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
