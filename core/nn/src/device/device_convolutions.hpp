#pragma once

#include "device_tensor.hpp"

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

namespace EigenSinn {

#ifdef __CUDACC__
  template<typename Scalar, int Layout = ColMajor>
  __global__ void dilate_tensor_kernel(long batches, long channels, long height, long width, long dilation,
    TensorView<Scalar, 4, Layout> dilated, TensorView<Scalar, 4, Layout> tensor) {

    // batch
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    // channel
    int c = threadIdx.y + blockIdx.y * blockDim.y;

    if (b < batches && c < channels) {
      for (long h = 0; h < height; h++) {
        for (long w = 0; w < width; w++) {
          dilated(b, c, h * dilation, w * dilation) = tensor(b, c, h, w);
        }
      }
    }
  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void set_col_kernel(TensorView<Scalar, 4, Layout> padded, TensorView<Scalar, 2, Layout> output,
    long shift, long batches, long channels, long dilation, long kernel_width, long kernel_height) {

    // batch
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    // channel
    int c = threadIdx.y + blockIdx.y * blockDim.y;

    if (b < batches && c < channels) {
      long col = 0;
      long col_batch = shift + b;

      for (long h_kernel = h_im; h_kernel < h_im + kernel_height; h_kernel += dilation) {
        for (long w_kernel = w_im; w_kernel < w_im + kernel_width; w_kernel += dilation, col++) {

          output(col, col_batch) = padded(b, c, h_kernel, w_kernel);
        }
      }

    }
  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void add_and_set_kernel(TensorView<Scalar, 4, Layout> out, TensorView<Scalar, 4, Layout> slice,
    long batches, long channels, long kernel_height, long kernel_width, long out_h, long out_w) {

    // batch
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    // channel
    int c = threadIdx.y + blockIdx.y * blockDim.y;

    if (b < batches && c < channels) {
      for (Index h = 0; h < kernel_height; h += dilation) {
        for (Index w = 0; w < kernel_width; w += dilation) {

          Index height_offset = h + out_h;
          Index width_offset = w + out_w;

          if (height_offset >= 0 && height_offset < out.dimension(2) && width_offset >= 0 && width_offset < out.dimension(3)) {
            out(b, c, height_offset, width_offset) += slice(b, c, h, w);
          }
        }
      }
    }
  }
#endif

  template<typename Scalar, int Layout = ColMajor, typename Device_>
  void setColFromSlice(Index batches, Index shift, const Index& channels, const Index& h_im,
    const Index& kernel_height, const Index& dilation,
    const Index& w_im, const Index& kernel_width,
    DeviceTensor<Device_, Scalar, 2, Layout>& output,
    DeviceTensor<Device_, Scalar, 4, Layout>& padded) {

#ifdef __CUDACC__
    if (std::is_same<Device_, GpuDevice>::value) {

      static dim3 block(BLOCK_SIZE, BLOCK_SIZE);
      static dim3 grid(getGridSize(batches, block.x), getGridSize(channels, block.y));

      set_col_kernel<Scalar, ColMajor> << <grid, block >> > (*padded, *output, shift,
        batches, channels, dilation, kernel_width, kernel_height);

      cudaDeviceSynchronize();
    }
    else {

#endif
      for (Index b = 0; b < batches; b++) {
        Index col = 0;
        Index col_batch = shift + b;

        for (Index c = 0; c < channels; c++) {
          for (Index h_kernel = h_im; h_kernel < h_im + kernel_height; h_kernel += dilation) {
            for (Index w_kernel = w_im; w_kernel < w_im + kernel_width; w_kernel += dilation, col++) {

              (*output)(col, col_batch) = (*padded)(b, c, h_kernel, w_kernel);
            }
          }
        }
      }
#ifdef __CUDACC__
    }
#endif
  }

  template<typename Scalar, int Layout, typename Device_>
  void addAndSet(const Index& batch_size, const Index& channels, const Index& kernel_height,
    int dilation, const Index& kernel_width, const Index& out_h, const Index& out_w,
    DeviceTensor<Device_, Scalar, 4, Layout>& out, DeviceTensor<Device_, Scalar, 4, Layout>& slice,
    Device_& device) {

#ifdef __CUDACC__
    if (std::is_same<Device_, GpuDevice>::value) {
      static dim3 block(BLOCK_SIZE, BLOCK_SIZE);
      static dim3 grid(getGridSize(batches, block.x), getGridSize(channels, block.y));

      add_and_set_kernel<Scalar, ColMajor> << <grid, block >> > (*out, *slice, batch_size, channels, kernel_height, kernel_width, out_h, out_w);

      cudaDeviceSynchronize();

    }
    else {
#endif
      for (Index b = 0; b < batch_size; b++) {
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
        }
      }

#ifdef __CUDACC__
    }
#endif
  }
} // EigenSinn