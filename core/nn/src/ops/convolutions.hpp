#pragma once

#include "device/device_tensor.hpp"
#include "device/device_convolutions.hpp"

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
  inline auto get_output_dimensions(const TensorView<Scalar, Rank, Layout>& input, const array<Index, Rank> kernel_dims, const Padding2D& padding, const Index stride, const Index dilation, bool is_transposed = false) {

    assert(kernel_dims[(int)ImageDims::channel] == input.dimension((int)ImageDims::channel));
    assert(kernel_dims[(int)ImageDims::height] > 0 && kernel_dims[(int)ImageDims::width] > 0);

    DSizes<Index, Rank> out_dims;

    Index conv_multiplyier = is_transposed ? -1 : 1;
    Index pad_height = conv_multiplyier * 2 * padding.first;
    Index pad_width = conv_multiplyier * 2 * padding.second;

    out_dims[(int)ImageDims::batch] = input.dimension((int)ImageDims::batch);
    out_dims[(int)ImageDims::channel] = kernel_dims[(int)ImageDims::batch];

    if (!is_transposed) {
      // see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
      out_dims[(int)ImageDims::height] = (input.dimension((int)ImageDims::height) + pad_height - dilation * (kernel_dims[(int)ImageDims::height] - 1) - 1) / stride + 1;
      out_dims[(int)ImageDims::width] = (input.dimension((int)ImageDims::width) + pad_width - dilation * (kernel_dims[(int)ImageDims::width] - 1) - 1) / stride + 1;
    }
    // see https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    else {
      out_dims[(int)ImageDims::height] = input.dimension((int)ImageDims::height) * stride + pad_height - dilation * (kernel_dims[(int)ImageDims::height] - 1) + 1;
      out_dims[(int)ImageDims::width] = input.dimension((int)ImageDims::width) * stride + pad_width - dilation * (kernel_dims[(int)ImageDims::width] - 1) + 1;

    }
    return out_dims;


  }

  // NCHW format
  template <typename Scalar, int Rank = 4, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto im2col(const DeviceTensor<Device_, Scalar, Rank, Layout>& input, const DSizes<Index, Rank>& kernel_dims, const Padding2D& padding, Index stride, Index dilation) {

    auto out_dims = get_output_dimensions(*input, kernel_dims, padding, stride, dilation);
    // pad the tensor before we convolve

    auto padded_dims = get_padded_input_dims(*input, padding);
    DeviceTensor<Device_, Scalar, 4, Layout> padded(padded_dims);
    padded.view() = input->pad(pad2dim(padding));

    Index kernel_height = dilation * (kernel_dims[2] - 1) + 1;
    Index kernel_width = dilation * (kernel_dims[3] - 1) + 1;
    Index channels = kernel_dims[1];
    Index batches = padded.dimension(0);

    Index col_dim = channels * kernel_dims[2] * kernel_dims[3];
    array<Index, Rank> slice_dims = { batches, channels, kernel_height, kernel_width };

    // output second dimension is 
    // batch_size (input dim[0]) * how_many_convolution_locations there are
    int conv_locations = out_dims[3] * out_dims[2];

    DeviceTensor<Device_, Scalar, 2, Layout> output(col_dim, input.dimension(0) * conv_locations);
    Device_ device = output.get_device();

    // "move" along axis 1 (columns) and append converted portions.
    // each converted portion is a a batch of would-be convolution operations
    Index converted_portion = 0;
    for (Index h_im = 0; h_im + kernel_height <= padded_dims[2]; h_im += stride) {
      for (Index w_im = 0; w_im + kernel_width <= padded_dims[3]; converted_portion++, w_im += stride) {

        Index shift = converted_portion * batches;
        setColFromSlice(batches, shift, channels, h_im, kernel_height, dilation, w_im, kernel_width, output, padded);
      }
    }
    return output;
  }

  template <typename Scalar, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline auto dilate_tensor(DeviceTensor<Device_, Scalar, 4, Layout>& tensor, Index dilation) {

    if (dilation == 1) { return tensor; }

    auto dims = tensor.dimensions();
    Index tensor_height = dilation * (dims[2] - 1) + 1;
    Index tensor_width = dilation * (dims[3] - 1) + 1;

    DeviceTensor<Device_, Scalar, 4, Layout> dilated(dims[0], dims[1], tensor_height, tensor_width);
    dilated.setZero();

#ifdef __CUDACC__
    if (std::is_same<Device_, GpuDevice>::value) {
      static dim3 block(BLOCK_SIZE, BLOCK_SIZE);
      static dim3 grid(getGridSize(dims[0], block.x), getGridSize(dims[1], block.y));
      
      dilate_tensor_kernel<Scalar, ColMajor> <<< grid, block >>> (dims[0], dims[1], dims[2], dims[3], dilation, *dilated, *tensor);
      cudaDeviceSynchronize();
    }
    else {
#endif
      // TODO: CUDA kernel for dilation
      for (Index b = 0; b < dims[0]; b++) {
        for (Index c = 0; c < dims[1]; c++) {
          for (Index h = 0; h < dims[2]; h++) {
            for (Index w = 0; w < dims[3]; w++) {
              (*dilated)(b, c, h * dilation, w * dilation) = (*tensor)(b, c, h, w);
            }
          }
        }
      }
#ifdef __CUDACC__
    }
#endif
    return dilated;
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
  // Non-GPU version
  template <typename Scalar, int Layout, typename Device_>
  inline auto col2im(const DeviceTensor<Device_, Scalar, 2, Layout>& col,
    const array<Index, 4>& kernel_dims,
    const array<Index, 4>& orig_dims,
    const Padding2D& padding,
    int stride = 1, int dilation = 1) {

    // intermediate output: original dimensions padded
    Index channels = kernel_dims[1],
      width = orig_dims[3] + 2 * padding.first,
      batch_size = orig_dims[0];

    array<Index, 2> col_dims = col.dimensions();
    DeviceTensor<Device_, Scalar, 4, ColMajor> out(orig_dims);
    out.setZero();

    Index out_w = 0, out_h = 0;
    array<Index, 2> slice_starts = { 0, 0 };
    array<Index, 2> slice_offsets = { col.dimension(0), batch_size };
    array<Index, 4> rev_shape = { kernel_dims[3], kernel_dims[2], kernel_dims[1], batch_size };
    DeviceTensor<Device_, Scalar, 4, ColMajor> slice(batch_size, kernel_dims[1], kernel_dims[2], kernel_dims[3]);

    Device_ device = out.get_device();

    Index kernel_height = dilation * (kernel_dims[2] - 1) + 1;
    Index kernel_width = dilation * (kernel_dims[3] - 1) + 1;

    // loop over col's batch size at a time
    // figure where it goes into the output
    // memcpy with setValues
    // shuffle dims to batch_size, channels, height, width
    // unpad with slice
    for (Index i = 0; i < col_dims[1] / batch_size; i++, slice_starts[1] += batch_size) {

      // move to the next slice
      out_w = (stride * i) % (width - kernel_width + 1) - padding.first;
      out_h = (stride * i) / (width - kernel_width + 1) - padding.second;

      slice.view() = col->slice(slice_starts, slice_offsets).eval().reshape(rev_shape).shuffle(array<Index, 4>{3, 2, 1, 0});
      addAndSet<Scalar, Layout, Device_>(batch_size, channels, kernel_height, dilation, kernel_width, out_h, out_w, out, slice, device);
    }

    return out;
  }

  // NCHW format
  template <typename Scalar, Index Rank = 4, int Layout = ColMajor, typename Device_ = DefaultDevice>
  inline DeviceTensor<Device_, Scalar, 4, Layout> convolve(const DeviceTensor<Device_, Scalar, 4, Layout>& input,
    DeviceTensor<Device_, Scalar, 4, Layout>& kernel, const Padding2D& padding, Index stride, Index dilation) {

    //dimensions involved in the convolution. Channel dimension is also involved.
    array<Index, 3> dims({ (int)ImageDims::channel, (int)ImageDims::height, (int)ImageDims::width });

    assert(input.dimension((int)ImageDims::channel) == kernel.dimension((int)ImageDims::channel));

    // output dimensions
    DSizes<Index, Rank> out_dims = get_output_dimensions(*input, *kernel, padding, stride, dilation);

    // perform convolutiion with GEMM using im2col
    auto col_inputs = im2col<Scalar, Rank, Layout, Device_>(input, kernel.dimensions(), padding, stride, dilation);
    auto unf_kernel = unfold_kernel<Scalar, Layout, Device_>(kernel);

    ProductDims prod_dims = { IndexPair<int>(1,0) };
    DeviceTensor<Device_, float, 2> res(unf_kernel.dimension(0), col_inputs.dimension(1));
    res.view() = unf_kernel->contract(*col_inputs, prod_dims);

    // NCHW output tensor
    DeviceTensor<Device_, Scalar, 4, Layout> output(out_dims);
    output = fold_conv_res<Scalar, Layout, Device_>(res, out_dims);

    return output;
  }

  // pad unevenly in case of k % 2 == 1:
  // more zero's goes upfront
  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_same(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, Index stride, Index dilation) {
    int dim1 = kernel.dimension((int)ImageDims::height) - 1;
    int dim2 = kernel.dimension((int)ImageDims::width) - 1;

    assert(dim1 & 0x1 == 0);
    assert(dim2 & 0x1 == 0);

    Tensor<Scalar, Rank> output = convolve(input, kernel, { dim1 / 2, dim2 / 2 }, stride, dilation);
    return output;
  }

  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_valid(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, Index stride, Index dilation) {
    Tensor<Scalar, Rank> output = convolve(input, kernel, { 0, 0 }, stride, dilation);
    return output;
  }

  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_full(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, Index stride, Index dilation) {
    int dim1 = kernel.dimension((int)ImageDims::height) - 1;
    int dim2 = kernel.dimension((int)ImageDims::width) - 1;

    Tensor<Scalar, Rank> output = convolve(input, kernel, { dim1, dim2 }, stride, dilation);
    return output;
  }
} // namespace EigenSinn
