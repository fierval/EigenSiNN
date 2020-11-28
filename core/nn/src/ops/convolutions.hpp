#pragma once

#include "device/device_tensor.hpp"

#define MAX_PAD 1e6
using namespace Eigen;

namespace EigenSinn {

  inline Padding pad2dim(int dim1, int dim2, int dim1_1, int dim2_1) {

    assert(dim1 >= 0 && dim1 < MAX_PAD&& dim2 >= 0 && dim2 < MAX_PAD&& dim2_1 < MAX_PAD&& dim1_1 < MAX_PAD);

    Padding paddings;
    paddings[(int)ImageDims::batch] = std::make_pair(0, 0);
    paddings[(int)ImageDims::height] = std::make_pair(dim1, dim1_1);
    paddings[(int)ImageDims::width] = std::make_pair(dim2, dim2_1);
    paddings[(int)ImageDims::channel] = std::make_pair(0, 0);

    return paddings;
  }

  template <typename Scalar, Index Rank, int Layout = ColMajor>
  inline array<Index, Rank> get_padded_input_dims(const TensorView<Scalar, Rank, Layout>& t, const Padding2D& pad2d) {
    array<Index, Rank> out({ t.dimension(0), t.dimension(1), t.dimension(2) + 2 * pad2d.first, t.dimension(3) + 2 * pad2d.second });
    return out;
  }

  inline Padding pad2dim(const Padding2D& pad2d) {
    return pad2dim(pad2d.first, pad2d.second, pad2d.first, pad2d.second);
  }

  template <typename Scalar, Index Rank = 4, int Layout = ColMajor>
  inline auto get_output_dimensions(const TensorView<Scalar, Rank, Layout>& input, const array<Index, Rank> kernel_dims, const Padding2D& padding, const Index stride = 1) {

    assert(kernel_dims[(int)ImageDims::channel] == input.dimension((int)ImageDims::channel));
    assert(kernel_dims[(int)ImageDims::height] > 0 && kernel_dims[(int)ImageDims::width] > 0);

    DSizes<Index, Rank> out_dims;

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
  template <typename Scalar, Index Rank = 4, int Layout = ColMajor>
  inline auto get_output_dimensions(const TensorView<Scalar, Rank, Layout>& input, const TensorView<Scalar, Rank, Layout>& kernel, const Padding2D& padding, const Index stride = 1) {

    return get_output_dimensions(input, kernel.dimensions(), padding, stride);
  }

  // NCHW format
  //TODO: support stride
  template <typename Scalar, Index Rank = 4, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline DeviceTensor<Device_, Scalar, 4, Layout> convolve(const DeviceTensor<Device_, Scalar, 4, Layout>& input,
    DeviceTensor<Device_, Scalar, 4, Layout>& kernel, const Padding2D& padding, Index stride = 1) {

    //dimensions involved in the convolution. Channel dimension is also involved.
    array<Index, 3> dims({ (int)ImageDims::channel, (int)ImageDims::height, (int)ImageDims::width });

    assert(input.dimension((int)ImageDims::channel) == kernel.dimension((int)ImageDims::channel));

    // Pad if apropriate
    auto padded_dims = get_padded_input_dims(*input, padding);
    DeviceTensor<Device_, Scalar, 4, Layout> padded(padded_dims);
    padded.view() = input->pad(pad2dim(padding));

    // output dimensions
    DSizes<Index, Rank> out_dims = get_output_dimensions(*input, *kernel, padding, stride);

    // NCHW output tensor
    DeviceTensor<Device_, Scalar, 4, Layout> output(out_dims);

    for (int i = 0; i < kernel.dimension((int)ImageDims::batch); i++) {
      // convolve on 3 dimensions and set the channel dimension of the entire batch
      output->chip(i, (int)ImageDims::channel).device(output.get_device()) = padded->convolve(kernel->chip(i, (int)ImageDims::batch), dims).chip(0, (int)ImageDims::channel);
    }

    return output;
  }

  // NCHW format
  template <typename Scalar, int Rank = 4, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto im2col(const DeviceTensor<Device_, Scalar, Rank, Layout>& input, const DSizes<Index, Rank>& kernel_dims, const Padding2D& padding, Index stride = 1) {

    auto out_dims = get_output_dimensions(*input, kernel_dims, padding, stride);
    // pad the tensor before we convolve

    auto padded_dims = get_padded_input_dims(*input, padding);
    DeviceTensor<Device_, Scalar, 4, Layout> padded(padded_dims);
    padded.view() = input->pad(pad2dim(padding));

    Index col_dim = kernel_dims[1] * kernel_dims[2] * kernel_dims[3];
    array<Index, Rank> starts = { 0, 0, 0, 0 };
    array<Index, Rank> offsets = { padded.dimension(0), kernel_dims[1], kernel_dims[2], kernel_dims[3] };
    array<Index, Rank> slice_dims = { kernel_dims[3], kernel_dims[2], kernel_dims[1], padded.dimension(0) };

    array<int, 4> shuffle_dims = { 3, 2, 1, 0 };
    // output second dimension is 
    // batch_size (input dim[0]) * how_many_convolution_locations there are
    int conv_locations = out_dims[3] * out_dims[2];

    DeviceTensor<Device_, Scalar, 2, Layout> output(col_dim, input.dimension(0) * conv_locations);

    // "move" along axis 1 (columns) and append converted portions.
    // each converted portion is a a batch of would-be convolution operations
    Index converted_portion = 0;
    for (starts[2] = 0; starts[2] + offsets[2] <= padded.dimension(2); starts[2] += stride) {
      for (starts[3] = 0; starts[3] + offsets[3] <= padded.dimension(3); converted_portion++, starts[3] += stride) {

        DeviceTensor<Device_, Scalar, 4, Layout> cur_slice(slice_dims);
        cur_slice.view() = padded->slice(starts, offsets).eval().shuffle(shuffle_dims);

        DeviceTensor<Device_, Scalar, 2, Layout> flat_slice(col_dim, padded.dimension(0));
        flat_slice.view() = cur_slice->reshape(DSizes<Index, 2>{ col_dim, padded.dimension(0) });

        int shift = converted_portion * flat_slice.dimension(1);
        for (Index i = 0; i < flat_slice.dimension(1); i++) {
          // REVIEW: This needs to be properly expressed through the DeviceTensor object
          output->chip(shift + i, 1).device(output.get_device()) = flat_slice->chip(i, 1);
        }
      }

    }
    return output;
  }

  // return kernel representation for GEMM with 
  // im2col representation of the conv layer
  template <typename Scalar, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto unfold_kernel(DeviceTensor<Device_, Scalar, 4, Layout>& kernel) {

    auto dims = kernel.dimensions();
    Index col_channels = dims[1] * dims[2] * dims[3];

    DeviceTensor<Device_, Scalar, 2, Layout> flat_kernel(dims[0], col_channels);
    flat_kernel.view() = kernel->shuffle(array<Index, 4>{ 0, 3, 2, 1 }).reshape(array<Index, 2>{dims[0], col_channels});
    return flat_kernel;
  }

  // convert back to the [b, c, h, w] kernel representation
  template <typename Scalar, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto fold_kernel(DeviceTensor<Device_, Scalar, 2, Layout>& kernel_col, const array<Index, 4>& expected_dims) {

    assert(expected_dims[0] == kernel_col.dimension(0));
    assert(expected_dims[1] * expected_dims[2] * expected_dims[3] == kernel_col.dimension(1));

    DeviceTensor<Device_, Scalar, 4, Layout> out(expected_dims);
    out.view() = kernel_col->reshape(array<Index, 4>{ expected_dims[0], expected_dims[3], expected_dims[2], expected_dims[1] })
      .shuffle(array<Index, 4>{0, 3, 2, 1});;

    return out;
  }

  // return output layer representation for GEMM with 
  // im2col representation of the conv layer
  // final dimensions should be the same as those of FX in col form:
  // F: [Bf x C * Hf * Wf], X: [C * Hf * Wf x B * Ho * Wo] -> [Bf X B * Ho * Wo], 
  // Resulting unrolled dimensions: [B, Bf, Ho, Wo], Bf = new C
  template <typename Scalar, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto unfold_conv_res(const DeviceTensor<Device_, Scalar, 4, Layout>& layer) {

    auto dims = layer.dimensions();
    Index col_channels = dims[0] * dims[2] * dims[3]; // B * Ho * Wo

    DeviceTensor<Device_, Scalar, 2, Layout> flat_layer(dims[1], col_channels);
    flat_layer.view() = layer->shuffle(array<Index, 4>{1, 0, 3, 2}).reshape(array<Index, 2>{dims[1], col_channels});
    return flat_layer;
  }


  // returns the convolution result in its "normal" form
  // assuming we have just performed a convolution operation.
  // e.g. t (*) k = r, t: [2, 3, 4, 4], k: [5, 3, 3, 3], r: [2, 5, 2, 2]
  // r in im2col form will be [5, 8]
  template <typename Scalar, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto fold_conv_res(const DeviceTensor<Device_, Scalar, 2, Layout>& conv_res, const array<Index, 4>& expected_dims) {

    assert(expected_dims[1] == conv_res.dimension(0));
    assert(expected_dims[0] * expected_dims[2] * expected_dims[3] == conv_res.dimension(1));

    DeviceTensor<Device_, Scalar, 4, Layout> out(expected_dims);
    out.view() = conv_res->reshape(array<Index, 4>{ expected_dims[1], expected_dims[0], expected_dims[3], expected_dims[2] }).shuffle(array<Index, 4>{1, 0, 3, 2});
    return out;
  }

  // NOT the reverse of im2col.
  // Used to fold the backward pass result of dX/dL. 
  // We still slide the kernel window over the 2d representation, 
  // Adding contributions of each folded slice to the result
  template <typename Scalar, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto col2im(const DeviceTensor<Device_, Scalar, 2, Layout>& col,
    const array<Index, 4>& kernel_dims,
    const array<Index, 4>& orig_dims,
    const Padding2D& padding,
    int stride = 1) {

    // intermediate output: original dimensions padded
    Index channels = kernel_dims[1],
      height = orig_dims[2] + 2 * padding.first,
      width = orig_dims[3] + 2 * padding.second,
      batch_size = orig_dims[0];

    array<Index, 2> col_dims = col.dimensions();
    DeviceTensor<Device_, Scalar, 4, Layout> out(batch_size, channels, height, width);
    out.setZero();

    Index out_w = 0, out_h = 0;
    array<Index, 2> slice_starts = { 0, 0 };
    array<Index, 2> slice_offsets = { col.dimension(0), batch_size };
    array<Index, 4> rev_shape = { kernel_dims[3], kernel_dims[2], kernel_dims[1], batch_size };
    DeviceTensor<Device_, Scalar, 4, Layout> slice(batch_size, kernel_dims[1], kernel_dims[2], kernel_dims[3]);

    // loop over col's batch size at a time
      // figure where it goes into the output
      // memcpy with setValues
    // shuffle dims to batch_size, channels, height, width
    // unpad with slice
    for (Index i = 0; i < col_dims[1] / batch_size; i++, slice_starts[1] += batch_size) {

      slice.view() = col->slice(slice_starts, slice_offsets).eval().reshape(rev_shape).shuffle(array<Index, 4>{3, 2, 1, 0});

      for (Index b = 0; b < batch_size; b++) {
        for (Index c = 0; c < channels; c++) {
          for (Index h = 0; h < kernel_dims[2]; h++) {
            for (Index w = 0; w < kernel_dims[3]; w++) {
              out->operator()(b, c, h + out_h, w + out_w) += slice->operator()(b, c, h, w);
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
    DeviceTensor<Device_, Scalar, 4, Layout> output(unpad_offsets);
    output.view() = out->slice(unpad_starts, unpad_offsets);
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

    Tensor<Scalar, Rank> output = convolve(input, kernel, { dim1 / 2, dim2 / 2 });
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
