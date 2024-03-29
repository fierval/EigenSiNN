#pragma once

#include "device/device_convolutions.hpp"
#include "helpers/conv_params_bag.hpp"

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

  template <typename Scalar, Index Rank, int Layout = RowMajor>
  inline array<Index, Rank> get_padded_input_dims(const TensorView<Scalar, Rank, Layout>& t, const Padding2D& pad2d) {
    array<Index, Rank> out({ t.dimension(0), t.dimension(1), t.dimension(2) + 2 * pad2d.first, t.dimension(3) + 2 * pad2d.second });
    return out;
  }

  inline Padding pad2dim(const Padding2D& pad2d) {
    return pad2dim(pad2d.first, pad2d.second, pad2d.first, pad2d.second);
  }

  // NCHW format
  template <typename Scalar, int Rank = 4, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline DeviceTensor<Scalar, 2, Device_, Layout> im2col(const DeviceTensor<Scalar, Rank, Device_, Layout>& input, ConvolutionParams<Rank> params) {

    int dilation = params.dilation, stride = params.stride;
    auto padding = params.padding;
    auto out_dims = params.output_dims();
    auto kernel_dims = params.kernel_dims;

    long kernel_height = params.dilated_kernel_height;
    long kernel_width = params.dilated_kernel_width;
    long channels = kernel_dims[1];
    long batches = params.orig_dims()[0];

    long col_dim = channels * kernel_dims[2] * kernel_dims[3];

    // output second dimension is 
    // batch_size (input dim[0]) * how_many_convolution_locations there are
    int conv_locations = out_dims[3] * out_dims[2];

    DeviceTensor<Scalar, 2, Device_, Layout> output(col_dim, input.dimension(0) * conv_locations);

    int out_width = out_dims[(int)ImageDims::width];

#ifndef __CUDACC__
    if (is_cpu(input.device())) {
      std::vector<long> h_im_range = params.h_im_range, w_im_range = params.w_im_range;

      std::for_each(std::execution::par_unseq, h_im_range.begin(), h_im_range.end(), [&](auto h_im) {
        std::for_each(std::execution::par_unseq, w_im_range.begin(), w_im_range.end(), [&](auto w_im) {

          setColFromSlice(batches, padding, channels, stride, dilation, h_im, w_im, kernel_height, kernel_width, output, input, out_width);
          });
        });
    }
#else
    if (std::is_same<Device_, GpuDevice>::value) {
      static dim3 block(BLOCK_SIZE, BLOCK_SIZE);
      dim3 grid(getGridSize(out_dims[(int)ImageDims::height], block.x), getGridSize(out_dims[(int)ImageDims::width], block.y));
      Device_ device = output.device();

      int out_height = out_dims[(int)ImageDims::height];

      set_col_kernel<Scalar, Layout> << <grid, block, 0, device.stream() >> >
        (batches, padding, channels, kernel_height, kernel_width, stride, dilation, *output, *input, out_height, out_width);

      cudaDeviceSynchronize();
    }
#endif
    return std::move(output);
  }

  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline auto dilate_tensor(DeviceTensor<Scalar, 4, Device_, Layout>& tensor, Index dilation) {

    if (dilation == 1) { return tensor; }

    auto dims = tensor.dimensions();
    Index tensor_height = dilation * (dims[2] - 1) + 1;
    Index tensor_width = dilation * (dims[3] - 1) + 1;

    DeviceTensor<Scalar, 4, Device_, Layout> dilated(dims[0], dims[1], tensor_height, tensor_width);
    dilated.setZero();

#ifdef __CUDACC__
    if (std::is_same<Device_, GpuDevice>::value) {
      static dim3 block(BLOCK_SIZE, BLOCK_SIZE);
      dim3 grid(getGridSize(dims[0], block.x), getGridSize(dims[1], block.y));

      dilate_tensor_kernel<Scalar, Layout> << < grid, block, 0, tensor.device().stream() >> > (dims[0], dims[1], dims[2], dims[3], dilation, *dilated, *tensor);
    }
#else
     if(is_cpu(tensor.device())) {
       for (Index b = 0; b < dims[0]; b++) {
         for (Index c = 0; c < dims[1]; c++) {
           for (Index h = 0; h < dims[2]; h++) {
             for (Index w = 0; w < dims[3]; w++) {
               (*dilated)(b, c, h * dilation, w * dilation) = (*tensor)(b, c, h, w);
             }
           }
         }
       }
      }
#endif
    return dilated;
  }

  // return kernel representation for GEMM with 
  // im2col representation of the conv layer
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline DeviceTensor<Scalar, 2, Device_, Layout> unfold_kernel(DeviceTensor<Scalar, 4, Device_, Layout>& kernel) {

    auto dims = kernel.dimensions();
    Index col_channels = dims[1] * dims[2] * dims[3];

    DeviceTensor<Scalar, 2, Device_, Layout> flat_kernel(dims[0], col_channels);

    // REVIEW: Looks like an Eigen bug in the shuffle()!!! It's incorrect when the layout is RowMajor
    if (Layout == RowMajor) {
      flat_kernel.view() = kernel->reshape(array<Index, 2>{dims[0], col_channels});
    }
    else {
      flat_kernel.view() = kernel->shuffle(array<Index, 4>{ 0, 3, 2, 1 }).reshape(array<Index, 2>{dims[0], col_channels});
    }
    return std::move(flat_kernel);
  }

  // convert back to the [b, c, h, w] kernel representation
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline auto fold_kernel(DeviceTensor<Scalar, 2, Device_, Layout>& kernel_col, const array<Index, 4>& expected_dims) {

    assert(expected_dims[0] == kernel_col.dimension(0));
    assert(expected_dims[1] * expected_dims[2] * expected_dims[3] == kernel_col.dimension(1));

    DeviceTensor<Scalar, 4, Device_, Layout> out(expected_dims);
    if (Layout != RowMajor) {
      out.view() = kernel_col->reshape(array<Index, 4>{ expected_dims[0], expected_dims[3], expected_dims[2], expected_dims[1] })
        .shuffle(array<Index, 4>{0, 3, 2, 1});
    } else {
      out.view() = kernel_col->reshape(array<Index, 4>{ expected_dims[0], expected_dims[1], expected_dims[2], expected_dims[3] });
    }

    return std::move(out);
  }

  // return output layer representation for GEMM with 
  // im2col representation of the conv layer
  // final dimensions should be the same as those of FX in col form:
  // F: [Bf x C * Hf * Wf], X: [C * Hf * Wf x B * Ho * Wo] -> [Bf X B * Ho * Wo], 
  // Resulting unrolled dimensions: [B, Bf, Ho, Wo], Bf = new C
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline auto unfold_conv_res(const DeviceTensor<Scalar, 4, Device_, Layout>& layer) {

    auto dims = layer.dimensions();
    Index col_channels = dims[0] * dims[2] * dims[3]; // B * Ho * Wo

    DeviceTensor<Scalar, 2, Device_, Layout> flat_layer(dims[1], col_channels);

    array<Index, 4> shuffle_dims = Layout != RowMajor ? array<Index, 4>{1, 0, 3, 2} : array<Index, 4>{1, 2, 3, 0};

    flat_layer.view() = layer->shuffle(shuffle_dims).reshape(array<Index, 2>{dims[1], col_channels});
    return std::move(flat_layer);
  }


  // returns the convolution result in its "normal" form
  // assuming we have just performed a convolution operation.
  // e.g. t (*) k = r, t: [2, 3, 4, 4], k: [5, 3, 3, 3], r: [2, 5, 2, 2]
  // r in im2col form will be [5, 8]
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline auto fold_conv_res(DeviceTensor<Scalar, 2, Device_, Layout>& conv_res, const array<Index, 4>& expected_dims) {

    assert(expected_dims[1] == conv_res.dimension(0));
    assert(expected_dims[0] * expected_dims[2] * expected_dims[3] == conv_res.dimension(1));

    DeviceTensor<Scalar, 4, Device_, Layout> out(expected_dims);

    if (Layout != RowMajor) {
      out.view() = conv_res->reshape(array<Index, 4>{ expected_dims[1], expected_dims[0], expected_dims[3], expected_dims[2] }).shuffle(array<Index, 4>{1, 0, 3, 2});
    }
    else {
      out.view() = conv_res->reshape(array<Index, 4>{ expected_dims[1], expected_dims[2], expected_dims[3], expected_dims[0] }).shuffle(array<Index, 4>{3, 0, 1, 2});
    }

    return std::move(out);
  }

  // NOT the reverse of im2col.
  // Used to fold the backward pass result of dX/dL. 
  // We still slide the kernel window over the 2d representation, 
  // Adding contributions of each folded slice to the result
  // Non-GPU version
  // TODO: is_dilated is a hack: we handle dilations by changing the kernel which means that dilation is already taken into account in most cases
  template <typename Scalar, int Layout, typename Device_>
  inline void col2im(const DeviceTensor<Scalar, 2, Device_, Layout>& col, DeviceTensor<Scalar, 4, Device_, Layout>& out, const ConvolutionParams<4>& params, bool is_dilated = false) {

    array<Index, 2> col_dims = col.dimensions();
    DSizes<Index, 4> orig_dims = params.orig_dims();
    const DSizes<Index, 4> kernel_dims = params.dilated_kernel_dims;
    const int stride = params.stride;

    // REVIEW: dilation is set only if we have not already taken dilation into account
    const int dilation = is_dilated ? 1 : params.dilation;
    auto padding = params.padding;

    // intermediate output: original dimensions padded
    long channels = kernel_dims[1],
      padded_width = orig_dims[3] + 2 * padding.first,
      batch_size = orig_dims[0],
      num_batches = col_dims[1] / batch_size;

    Device_ device = out.device();
    out.setZero();

    Index kernel_height = params.dilated_kernel_height;
    Index kernel_width = params.dilated_kernel_width;

    // loop over col's batch size at a time
    // figure where it goes into the output
#ifndef __CUDACC__
    if (!std::is_same <Device_, GpuDevice>::value) {
      for (int batch = 0; batch < num_batches; batch++) {

        addAndSet<Scalar, Layout, Device_>(batch_size, batch, channels, stride, padding, dilation, kernel_height, kernel_width, padded_width, out, col);
      }
    }
#else
    if (std::is_same<Device_, GpuDevice>::value) {
      static int block(BLOCK_SIZE * BLOCK_SIZE);
      int grid(getGridSize(num_batches, block));

      add_and_set_kernel<Scalar, Layout> << < grid, block, 0, device.stream() >> >
        (batch_size, num_batches, channels, stride, padding, dilation, kernel_height, kernel_width, padded_width, *out, *col);

      cudaDeviceSynchronize();
    }
#endif
  }

  // NCHW format
  template <typename Scalar, Index Rank = 4, int Layout = RowMajor, typename Device_ = ThreadPoolDevice>
  inline DeviceTensor<Scalar, 4, Device_, Layout> convolve(const DeviceTensor<Scalar, 4, Device_, Layout>& input,
    DeviceTensor<Scalar, 4, Device_, Layout>& kernel, const ConvolutionParams<Rank>& params) {

    assert(input.dimension((int)ImageDims::channel) == kernel.dimension((int)ImageDims::channel));

    // output dimensions
    DSizes<Index, Rank> out_dims = params.output_dims();

    // perform convolutiion with GEMM using im2col
    auto col_inputs = im2col<Scalar, Rank, Device_, Layout>(input, params);
    auto unf_kernel = unfold_kernel<Scalar, Device_, Layout>(kernel);

    ProductDims prod_dims = { IndexPair<int>(1,0) };
    DeviceTensor<float, 2, Device_, Layout> res(unf_kernel.dimension(0), col_inputs.dimension(1));
    res.view() = unf_kernel->contract(*col_inputs, prod_dims);

    // NCHW output tensor
    return std::move(fold_conv_res<Scalar, Device_>(res, out_dims));
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
    return std::move(output);
  }

  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_valid(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, Index stride, Index dilation) {
    Tensor<Scalar, Rank> output = convolve(input, kernel, { 0, 0 }, stride, dilation);
    return std::move(output);
  }

  template <typename Scalar, Index Rank = 4>
  inline Tensor<Scalar, Rank> convolve_full(Tensor<Scalar, Rank>& input, Tensor<Scalar, Rank>& kernel, Index stride, Index dilation) {
    int dim1 = kernel.dimension((int)ImageDims::height) - 1;
    int dim2 = kernel.dimension((int)ImageDims::width) - 1;

    Tensor<Scalar, Rank> output = convolve(input, kernel, { dim1, dim2 }, stride, dilation);
    return std::move(output);
  }
} // namespace EigenSinn
