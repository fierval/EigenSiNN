#pragma once

#include "device_helpers.hpp"
#define MAXPOOL_BLOCK_SIZE 16

namespace EigenSinn {

#ifdef __CUDACC__
  template<typename Scalar1, typename Scalar2>
  __global__ void set_from_tuple_kernel(Scalar1* dest1, Scalar2* dest2, Tuple<Scalar1, Scalar2>* src) {
    *dest1 = src->first;
    *dest2 = src->second;
  }


    template<typename Scalar, int Layout = ColMajor>
    __global__ void maxpool_dinput_tensor_kernel4d (
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
        
        long idx_output = to_flat_dim<long, 4, Layout>(out_dims, {batch, channel, (long)output_pos.y + unrolled_dim[2], (long)output_pos.x + unrolled_dim[3]});
        long idx_grad = to_flat_dim<long, 4, Layout>(mask_dims, {batch, channel, (long)grad_starts.y, (long)grad_starts.x});

        output.data()[idx_output] += dout.data()[idx_grad];
      }

  }

    template<typename Scalar, int Layout = ColMajor>
    __global__ void maxpool_dinput_tensor_kernel2d(
      TensorView<Scalar, 2, Layout> output, TensorView<Scalar, 2, Layout> dout, TensorView<Index, 2, Layout> mask
      , long batches, long grad_starts, long extents, long output_pos) {

      int batch = threadIdx.x + blockIdx.x * blockDim.x;

      if (batch < batches) {

        array<long, 2> pool_window_dims{ batches, (long)extents};

        // TODO: for some reason creating Index-type (long long) arrays crashes the kernel
        // so convering everything to "long"
        auto dims = mask.dimensions();
        auto mask_dims = array<long, 2>{(long)dims[0], (long)dims[1]};

        dims = output.dimensions();
        auto out_dims = array<long, 2>{(long)dims[0], (long)dims[1]};

        auto idx = to_flat_dim<long, 2, Layout>(mask_dims, { batch, (long)grad_starts});
        auto idx_flat = mask.data()[idx];

        auto idx_col = from_flat_dim<long, 2, ColMajor>(pool_window_dims, idx_flat)[1];

        long idx_output = to_flat_dim<long, 2, Layout>(out_dims, { batch, (long)output_pos + idx_col});
        long idx_grad = to_flat_dim<long, 2, Layout>(mask_dims, { batch, (long)grad_starts});

        output.data()[idx_output] += dout.data()[idx_grad];
      }

    }

#endif  

  template<typename Scalar1, typename Scalar2, int Rank, int Layout, typename Device_>
  void set_from_tuple(TensorView<Scalar1, Rank, Layout>& dest1,
    TensorView<Scalar2, Rank, Layout>& dest2,
    const array<Index, Rank>& dest_offset,
    const TensorView<Tuple<Index, Scalar2>, Rank / 2, Layout>& src,
    const array<Index, Rank / 2> src_offset, const Device_& device) {

#ifdef __CUDACC__
    if (std::is_same<Device_, GpuDevice>::value) {
      //// launch kernel
      auto idx_dest_offset = Layout == ColMajor ? dest1.dimensions().IndexOfColMajor(dest_offset) : dest1.dimensions().IndexOfRowMajor(dest_offset);
      auto idx_src_offset = Layout == ColMajor ? src.dimensions().IndexOfColMajor(src_offset) : src.dimensions().IndexOfRowMajor(src_offset);

      Tuple<Index, Scalar2>* ptr_src = &src.data()[idx_src_offset];
      Scalar1* ptr_dest1 = &dest1.data()[idx_dest_offset];
      Scalar2* ptr_dest2 = &dest2.data()[idx_dest_offset];

      set_from_tuple_kernel<Scalar1, Scalar2> << <1, 1 >> > (ptr_dest1, ptr_dest2, ptr_src);
      cudaDeviceSynchronize();
    }
    else {
#endif
      dest1(dest_offset) = src(src_offset).first;
      dest2(dest_offset) = src(src_offset).second;
#ifdef __CUDACC__
    }
#endif

  }

}