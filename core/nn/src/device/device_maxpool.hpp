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
  __global__ void maxpool_dinput_kernel4d(Scalar* output, Scalar* dout, Index* mask, Index batches, Index channels,
    dim3 in_size, dim3 out_size, dim3 grad_starts, dim3 extents, dim3 output_pos) {

    Index batch = threadIdx.x + blockIdx.x * blockDim.x;
    Index channel = threadIdx.y + blockIdx.y * blockDim.y;

    if (batch < batches && channel < channels) {
      array<Index, 4> mask_dims{ batches, channels, in_size.y, in_size.x };
      array<Index, 4> pool_window_dims{ batches, channels, extents.y, extents.x };
      array<Index, 4> out_gradient_dims{ batches, channels, out_size.y, out_size.x };

      Index idx_flat = mask[to_flat_dim<4, Layout>(mask_dims, array<Index, 4>{batch, channel, grad_starts.y, grad_starts.x})];
      array<Index, 4> unrolled_dim = from_flat_dim<4, ColMajor>(pool_window_dims, idx_flat);

      Index idx_output = to_flat_dim<4, Layout>(mask_dims, array<Index, 4>{batch, channel, output_pos.y + unrolled_dim[2], output_pos.x + unrolled_dim[3]});
      Index idx_grad = to_flat_dim<4, Layout>(out_gradient_dims, array<Index, 4>{batch, channel, grad_starts.y, grad_starts.x});

      output[idx_output] += dout[idx_grad];
    }
  }

    template<typename Scalar, int Layout = ColMajor>
    __global__ void maxpool_dinput_tensor_kernel4d(TensorView<Scalar, 4, Layout> output, TensorView<Scalar, 4, Layout> dout, TensorView<Index, 4, Layout> mask, dim3 grad_starts, dim3 extents, dim3 output_pos) {

      Index batch = threadIdx.x + blockIdx.x * blockDim.x;
      Index channel = threadIdx.y + blockIdx.y * blockDim.y;

      Index batches = dout.dimension(0), channels = dout.dimension(1);
      if (batch < batches && channel < channels) {

        array<Index, 4> pool_window_dims{ batches, channels, extents.y, extents.x };

        Index idx_flat = mask(batch, channel, grad_starts.y, grad_starts.x);
        array<Index, 4> unrolled_dim = from_flat_dim<4, ColMajor>(pool_window_dims, idx_flat);

        output(batch, channel, output_pos.y + unrolled_dim[2], output_pos.x + unrolled_dim[3]) += dout(batch, channel, grad_starts.y, grad_starts.x);
      }

  }
#endif  

  template<typename Scalar1, typename Scalar2, Index Rank, int Layout, typename Device_>
  void set_from_tuple(TensorView<Scalar1, Rank, Layout>& dest1,
    TensorView<Scalar2, Rank, Layout>& dest2,
    const array<Index, Rank>& dest_offset,
    const TensorView<Tuple<Scalar1, Scalar2>, Rank / 2, Layout>& src,
    const array<Index, Rank / 2> src_offset, const Device_& device) {

#ifdef __CUDACC__
    if (std::is_same<Device_, GpuDevice>::value) {
      //// launch kernel
      auto idx_dest_offset = Layout == ColMajor ? dest1.dimensions().IndexOfColMajor(dest_offset) : dest1.dimensions().IndexOfRowMajor(dest_offset);
      auto idx_src_offset = Layout == ColMajor ? src.dimensions().IndexOfColMajor(src_offset) : src.dimensions().IndexOfRowMajor(src_offset);

      Tuple<Scalar1, Scalar2>* ptr_src = &src.data()[idx_src_offset];
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