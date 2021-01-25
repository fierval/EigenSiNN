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

          auto flat_out_dims = dimensions_cast<long>(dilated.dimensions());
          auto flat_inp_dims = dimensions_cast<long>(tensor.dimensions());

          long idx_output = to_flat_dim<long, 4, Layout>(flat_out_dims, { b, c, h * dilation, w * dilation });
          long idx_inp = to_flat_dim<long, 4, Layout>(flat_inp_dims, { b, c, h, w });

          dilated.data()[idx_output] = tensor.data()[idx_inp];
        }
      }
    }
  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void set_col_kernel(TensorView<Scalar, 4, Layout> padded, TensorView<Scalar, 2, Layout> output,
    long shift, long batches, long channels, long dilation, long kernel_width, long kernel_height, long h_im, long w_im) {

    // batch
    long b = threadIdx.x + blockIdx.x * blockDim.x;
    // channel
    long c = threadIdx.y + blockIdx.y * blockDim.y;

    if (b < batches && c < channels) {

      auto flat_out_dims = dimensions_cast<long>(output.dimensions());
      auto flat_inp_dims = dimensions_cast<long>(padded.dimensions());

      // need to recover the actual kernel width & height
      // in order to figure out exactly where in the output the input value belongs
      long undilated_width = (kernel_width - 1) / dilation + 1;
      long undilated_height = (kernel_height - 1) / dilation + 1;
      
      long col = c * undilated_height * undilated_width;
      long col_batch = shift + b;

      for (long h_kernel = h_im; h_kernel < h_im + kernel_height; h_kernel += dilation) {
        for (long w_kernel = w_im; w_kernel < w_im + kernel_width; w_kernel += dilation, col++) {

          
          long idx_output = to_flat_dim<long, 2, Layout>(flat_out_dims, {col, col_batch});
          long idx_inp = to_flat_dim<long, 4, Layout>(flat_inp_dims, { b, c, h_kernel, w_kernel});

          output.data()[idx_output] = padded.data()[idx_inp];
        }
      }

    }
  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void add_and_set_kernel(TensorView<Scalar, 4, Layout> out, TensorView<Scalar, 4, Layout> slice,
    long batches, long channels, long kernel_height, long kernel_width, long out_h, long out_w, long dilation) {

    // batch
    long b = threadIdx.x + blockIdx.x * blockDim.x;
    // channel
    long c = threadIdx.y + blockIdx.y * blockDim.y;

    if (b < batches && c < channels) {

      for (long h = 0; h < kernel_height; h += dilation) {
        for (long w = 0; w < kernel_width; w += dilation) {

          long height_offset = h + out_h;
          long width_offset = w + out_w;

          if (height_offset >= 0 && height_offset < out.dimension(2) && width_offset >= 0 && width_offset < out.dimension(3)) {
            auto flat_out_dims = dimensions_cast<long>(out.dimensions());
            auto flat_inp_dims = dimensions_cast<long>(slice.dimensions());

            long idx_output = to_flat_dim<long, 4, Layout>(flat_out_dims, { b, c, height_offset, width_offset });
            long idx_inp = to_flat_dim<long, 4, Layout>(flat_inp_dims, { b, c, h, w });

            out.data()[idx_output] += slice.data()[idx_inp];
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
        batches, channels, dilation, kernel_width, kernel_height, h_im, w_im);

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
      static dim3 grid(getGridSize(batch_size, block.x), getGridSize(channels, block.y));

      add_and_set_kernel<Scalar, ColMajor> << <grid, block >> > (*out, *slice, 
        batch_size, channels, kernel_height, kernel_width, out_h, out_w, dilation);

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