#pragma once

#include "device_helpers.hpp"

namespace EigenSinn {

#ifdef __CUDACC__
  template<typename Scalar1, typename Scalar2>
  __global__ void set_from_tuple_kernel(Scalar1* dest1, Scalar2* dest2, Tuple<Scalar1, Scalar2>* src) {
    *dest1 = src->first;
    *dest2 = src->second;
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