#pragma once

#include "device_helpers.hpp"
#define MAXPOOL_BLOCK_SIZE 16

namespace EigenSinn {

#ifdef __CUDACC__

  template<typename Scalar, int Layout = ColMajor>
  __global__ void maxpool_set_values_kernel4d(
    TensorView<Scalar, 4, Layout> output, TensorView<Index, 4, Layout> mask, TensorView<Tuple<Index, Scalar>, 2, Layout> local_pool, 
      dim3 in_size, dim3 output_starts) {

    int batch = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = threadIdx.y + blockIdx.y * blockDim.y;
    
    int channels = in_size.y;
    int batches = in_size.x;

    if (batch < batches && channel < channels) {

      // TODO: for some reason creating Index-type (long long) arrays crashes the kernel
      // so convering everything to "long"
      auto dims = mask.dimensions();
      auto out_dims = array<long, 4>{(long)dims[0], (long)dims[1], (long)dims[2], (long)dims[3]};

      array<long, 2> local_dims{ batches, channels };

      auto idx_output = to_flat_dim<long, 4, Layout>(out_dims, { batch, channel, (long)output_starts.y, (long)output_starts.x });
      auto idx_tuple = to_flat_dim<long, 2, Layout>(local_dims, { batch, channel });

      mask.data()[idx_output] = local_pool.data()[idx_tuple].first;
      output.data()[idx_output] = local_pool.data()[idx_tuple].second;
    }

  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void maxpool_dinput_kernel4d(
    TensorView<Scalar, 4, Layout> output, TensorView<Scalar, 4, Layout> dout, TensorView<Index, 4, Layout> mask
    , long batches, long channels, dim3 grad_starts, dim3 extents, dim3 output_pos) {

    int batch = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = threadIdx.y + blockIdx.y * blockDim.y;

    if (batch < batches && channel < channels) {

      array<long, 4> pool_window_dims{ batches, channels, (long)extents.y, (long)extents.x };

      // TODO: for some reason creating Index-type (long long) arrays crashes the kernel
      // so convering everything to "long"
      auto dims = mask.dimensions();
      auto mask_dims = array<long, 4>{(long)dims[0], (long)dims[1], (long)dims[2], (long)dims[3]};

      dims = output.dimensions();
      auto out_dims = array<long, 4>{(long)dims[0], (long)dims[1], (long)dims[2], (long)dims[3]};

      auto idx = to_flat_dim<long, 4, Layout>(mask_dims, { batch, channel, (long)grad_starts.y, (long)grad_starts.x });
      auto idx_flat = mask.data()[idx];

      auto unrolled_dim = from_flat_dim<long, 4, ColMajor>(pool_window_dims, idx_flat);

      long idx_output = to_flat_dim<long, 4, Layout>(out_dims, { batch, channel, (long)output_pos.y + unrolled_dim[2], (long)output_pos.x + unrolled_dim[3] });
      long idx_grad = to_flat_dim<long, 4, Layout>(mask_dims, { batch, channel, (long)grad_starts.y, (long)grad_starts.x });

      output.data()[idx_output] += dout.data()[idx_grad];
    }
  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void maxpool_set_values_kernel2d(
    TensorView<Scalar, 2, Layout> output, TensorView<Index, 2, Layout> mask, TensorView<Tuple<Index, Scalar>, 1, Layout> local_pool,
        long batches, long output_starts) {

    int batch = threadIdx.x + blockIdx.x * blockDim.x;

    if (batch < batches) {

      // TODO: for some reason creating Index-type (long long) arrays crashes the kernel
      // so convering everything to "long"
      auto dims = mask.dimensions();
      auto out_dims = array<long, 2>{(long)dims[0], (long)dims[1]};

      auto idx_output = to_flat_dim<long, 2, Layout>(out_dims, { batch, output_starts});

      mask.data()[idx_output] = local_pool.data()[batch].first;
      output.data()[idx_output] = local_pool.data()[batch].second;
    }

  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void maxpool_dinput_kernel2d(
    TensorView<Scalar, 2, Layout> output, TensorView<Scalar, 2, Layout> dout, TensorView<Index, 2, Layout> mask
    , long batches, long grad_starts, long extents, long output_pos) {

    int batch = threadIdx.x + blockIdx.x * blockDim.x;

    if (batch < batches) {

      array<long, 2> pool_window_dims{ batches, (long)extents };

      // TODO: for some reason creating Index-type (long long) arrays crashes the kernel
      // so convering everything to "long"
      auto dims = mask.dimensions();
      auto mask_dims = array<long, 2>{(long)dims[0], (long)dims[1]};

      dims = output.dimensions();
      auto out_dims = array<long, 2>{(long)dims[0], (long)dims[1]};

      auto idx = to_flat_dim<long, 2, Layout>(mask_dims, { batch, (long)grad_starts });
      auto idx_flat = mask.data()[idx];

      auto idx_col = from_flat_dim<long, 2, ColMajor>(pool_window_dims, idx_flat)[1];

      long idx_output = to_flat_dim<long, 2, Layout>(out_dims, { batch, (long)output_pos + idx_col });
      long idx_grad = to_flat_dim<long, 2, Layout>(mask_dims, { batch, (long)grad_starts });

      output.data()[idx_output] += dout.data()[idx_grad];
    }

  }

#endif  
}