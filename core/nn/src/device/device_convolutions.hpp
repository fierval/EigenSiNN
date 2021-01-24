#pragma once

#include "device_tensor.hpp"
#define CONV_BLOCK_SIZE 16

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

namespace EigenSinn {

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

  template<typename Scalar, int Layout = ColMajor, typename Device_>
  void SetColFromSlice(Index batches, Index shift, const Index& channels, const Index& h_im,
    const Index& kernel_height, const Index& dilation,
    const Index& w_im, const Index& kernel_width,
    DeviceTensor<Device_, Scalar, 2, Layout>& output,
    DeviceTensor<Device_, Scalar, 4, Layout>& padded) {

#ifdef __CUDACC__
    if (std::is_same<Device_, GpuDevice>::value) {

      static dim3 block(CONV_BLOCK_SIZE, CONV_BLOCK_SIZE);
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

#ifdef __CUDACC__
  template<typename Scalar>
  __global__ void add_and_set_kernel(Scalar* dest, Scalar* src) {
    *dest += *src;
  }
#endif

  template<typename Scalar, Index Rank, int Layout, typename Device_>
  inline void add_and_set(TensorView<Scalar, Rank, Layout>& dest, const array<Index, Rank>& dest_offset,
    const TensorView<Scalar, Rank, Layout>& src, const array<Index, Rank>& src_offset, const Device_& device) {

#ifdef __CUDACC__
    if (std::is_same<Device_, GpuDevice>::value) {
      // launch kernel
      auto idx_dest_offset = Layout == ColMajor ? dest.dimensions().IndexOfColMajor(dest_offset) : dest.dimensions().IndexOfRowMajor(dest_offset);
      auto idx_src_offset = Layout == ColMajor ? src.dimensions().IndexOfColMajor(src_offset) : src.dimensions().IndexOfRowMajor(src_offset);

      Scalar* ptr_src = &src.data()[idx_src_offset];
      Scalar* ptr_dest = &dest.data()[idx_dest_offset];

      add_and_set_kernel<Scalar> << <1, 1 >> > (ptr_dest, ptr_src);
      cudaDeviceSynchronize();
    }
    else {
#endif
      dest(dest_offset) += src(src_offset);
#ifdef __CUDACC__
    }
#endif
  }

} // EigenSinn