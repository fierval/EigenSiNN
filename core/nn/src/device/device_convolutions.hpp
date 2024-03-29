#pragma once

#include "device_tensor.hpp"
#include <execution>

namespace EigenSinn {

#ifdef __CUDACC__
  template<typename Scalar, int Layout = RowMajor>
  __global__ void dilate_tensor_kernel(long batches, long channels, long height, long width, long dilation,
    TensorView<Scalar, 4, Layout> dilated, TensorView<Scalar, 4, Layout> tensor) {

    // batch
    long b = threadIdx.x + blockIdx.x * blockDim.x;
    // channel
    long c = threadIdx.y + blockIdx.y * blockDim.y;

    if (b < batches && c < channels) {
      for (long h = 0; h < height; h++) {
        for (long w = 0; w < width; w++) {

          auto flat_out_dims = dimensions_cast<long>(dilated.dimensions());
          auto flat_inp_dims = dimensions_cast<long>(tensor.dimensions());

          long idx_output = to_flat_dim<long, 4, Layout>(flat_out_dims, { b, c, h * dilation, w * dilation });
          long idx_inp = to_flat_dim<long, 4, Layout>(flat_inp_dims, { b, c, h, w });

          dilated.data()[idx_output] = tensor.data()[idx_inp];
        }
      }
    }
  }

  template<typename Scalar, int Layout = RowMajor>
  __global__ void set_col_kernel(long batches, Padding2D padding, int channels,
    int kernel_height, int kernel_width, int stride, int dilation,
    TensorView<Scalar, 2, Layout> output,
    TensorView<Scalar, 4, Layout> input, int out_height, int out_width) {

    // current height in terms of the output
    long cur_height = threadIdx.x + blockIdx.x * blockDim.x;
    // current width in terms of the output
    long cur_width = threadIdx.y + blockIdx.y * blockDim.y;

    // gate
    if (cur_height >= out_height || cur_width >= out_width) {
      return;
    }

    // place in the image where to put the kernel at the moment
    long h_im = cur_height * stride - padding.first;
    long w_im = cur_width * stride - padding.second;

    // shift into the output matrix columns where to start putting flattend data
    long shift = batches * (cur_height * out_width + cur_width);

    auto flat_out_dims = dimensions_cast<long>(output.dimensions());
    auto flat_inp_dims = dimensions_cast<long>(input.dimensions());

    for (long b = 0; b < batches; b++) {
      long col = 0;
      long col_batch = shift + b;

      for (long c = 0; c < channels; c++) {
        for (long h_kernel = h_im; h_kernel < h_im + kernel_height; h_kernel += dilation) {
          for (long w_kernel = w_im; w_kernel < w_im + kernel_width; w_kernel += dilation, col++) {

            long idx_output = to_flat_dim<long, 2, Layout>(flat_out_dims, { col, col_batch });

            if (h_kernel >= 0 && w_kernel >= 0 && h_kernel < flat_inp_dims[2] && w_kernel < flat_inp_dims[3]) {

              long idx_inp = to_flat_dim<long, 4, Layout>(flat_inp_dims, { b, c, h_kernel, w_kernel });
              output.data()[idx_output] = input.data()[idx_inp];
            }
            else {
              output.data()[idx_output] = Scalar(0);
            }
          }
        }
      }
    }
  }

  template<typename Scalar, int Layout = RowMajor>
  __global__ void add_and_set_kernel(long batch_size, long num_batches, long channels, int stride, Padding2D padding,
    int dilation, long kernel_height, long kernel_width, long padded_width,
    TensorView<Scalar, 4, Layout> out, TensorView<Scalar, 2, Layout> col) {

    long batch = threadIdx.x + blockDim.x * blockIdx.x;

    if (batch >= num_batches) {
      return;
    }

    long out_w = (stride * batch) % (padded_width - kernel_width + 1) - padding.first;
    long out_h = (stride * batch) / (padded_width - kernel_width + 1) - padding.second;
    long col_start = batch * batch_size;

    // indices into the 2d "col" tensor
    long idx_col = col_start;
    long idx_row = 0;

    auto flat_out_dims = dimensions_cast<long>(out.dimensions());
    auto flat_inp_dims = dimensions_cast<long>(col.dimensions());

    for (long b = 0; b < batch_size; b++, idx_row = 0, idx_col++) {
      for (long c = 0; c < channels; c++) {
        for (long h = 0; h < kernel_height; h += dilation) {
          for (long w = 0; w < kernel_width; w += dilation, idx_row++) {

            long height_offset = h + out_h;
            long width_offset = w + out_w;

            if (height_offset >= 0 && height_offset < out.dimension(2) && width_offset >= 0 && width_offset < out.dimension(3)) {

              long idx_output = to_flat_dim<long, 4, Layout>(flat_out_dims, { b, c, height_offset, width_offset });
              long idx_inp = to_flat_dim<long, 2, Layout>(flat_inp_dims, { idx_row, idx_col });

              atomicAdd(&out.data()[idx_output], col.data()[idx_inp]);

            }
          }
        }
      }
    }

  }

#endif

  template<typename Scalar, int Layout = RowMajor, typename Device_>
  void setColFromSlice(Index batches, Padding2D& padding, const Index& channels, const int& stride, const Index& dilation,
    const Index& h_im, const Index& w_im,
    const Index& kernel_height, const Index& kernel_width,
    DeviceTensor<Scalar, 2, Device_, Layout>& output,  const DeviceTensor<Scalar, 4, Device_, Layout>& input, int output_width) {


    long shift = batches * ((h_im + padding.first) / stride * output_width + (w_im + padding.second) / stride);

    for (Index b = 0; b < batches; b++) {
      Index col = 0;
      Index col_batch = shift + b;

      for (Index c = 0; c < channels; c++) {
        for (Index h_kernel = h_im; h_kernel < h_im + kernel_height; h_kernel += dilation) {
          for (Index w_kernel = w_im; w_kernel < w_im + kernel_width; w_kernel += dilation, col++) {

            if (h_kernel >= 0 && w_kernel >= 0 && h_kernel < input.dimension(2) && w_kernel < input.dimension(3)) {
              (*output)(col, col_batch) = (*input)(b, c, h_kernel, w_kernel);
            }
            else {
              (*output)(col, col_batch) = Scalar(0);
            }
          }
        }
      }
    }
  }

  template<typename Scalar, int Layout, typename Device_>
  void addAndSet(const long& batch_size, const int batch, const long& channels, const int stride, const Padding2D& padding,
    int dilation, const long& kernel_height, const long& kernel_width, const long& padded_width,
    DeviceTensor<Scalar, 4, Device_, Layout>& out, const DeviceTensor<Scalar, 2, Device_, Layout>& col) {

    // move to the next slice
    long out_w = (stride * batch) % (padded_width - kernel_width + 1) - padding.first;
    long out_h = (stride * batch) / (padded_width - kernel_width + 1) - padding.second;
    long col_start = batch * batch_size;

    // indices into the 2d "col" tensor
    long idx_col = col_start;
    long idx_row = 0;

    for (Index b = 0; b < batch_size; b++, idx_row = 0, idx_col++) {
      for (Index c = 0; c < channels; c++) {
        for (Index h = 0; h < kernel_height; h += dilation) {
          for (Index w = 0; w < kernel_width; w += dilation, idx_row++) {

            Index height_offset = h + out_h;
            Index width_offset = w + out_w;

            if (height_offset >= 0 && height_offset < out.dimension(2) && width_offset >= 0 && width_offset < out.dimension(3)) {

              (*out)(b, c, height_offset, width_offset) += (*col)(idx_row, idx_col);
            }
          }
        }
      }
    }
  }

#ifndef __CUDACC__
  template<typename Scalar, int Layout, typename Device_>
  void addAndSet_CPU(const Index& batch_size, const Index& channels, const Index& kernel_height,
    int dilation, const Index& kernel_width, const Index& out_h, const Index& out_w,
    DeviceTensor<Scalar, 4, Device_, Layout>& out, DeviceTensor<Scalar, 4, Device_, Layout>& slice,
    Device_& device) {

    if (!std::is_same<Device_, ThreadPoolDevice>::value && !std::is_same<Device_, DefaultDevice>::value) {
      throw std::invalid_argument("CPU device required");
    }

    std::vector<Index> batch_vector(batch_size);
    std::iota(batch_vector.begin(), batch_vector.end(), 0);

    std::for_each(std::execution::par_unseq, batch_vector.begin(), batch_vector.end(), [&](auto b) {
      for (Index c = 0; c < channels; c++) {
        for (Index h = 0; h < kernel_height; h += dilation) {
          for (Index w = 0; w < kernel_width; w += dilation) {

            Index height_offset = h + out_h;
            Index width_offset = w + out_w;

            if (height_offset >= 0 && height_offset < out.dimension(2) && width_offset >= 0 && width_offset < out.dimension(3)) {
              (*out)(b, c, height_offset, width_offset) += (*slice)(b, c, h, w);
            }
          }
        }
      }});
  }
#endif
} // EigenSinn