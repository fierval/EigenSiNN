#pragma once
#include "ops/opsbase.hpp"

using namespace Eigen;
using std::unique_ptr;
using std::make_unique;

// block size for CUDA operations
#define BLOCK_SIZE 16

namespace EigenSinn {

  template<typename Device_, typename Scalar, Index Rank>
  inline void free(TensorView<Scalar, Rank>& t, Device_& device) {

    if (t.data() != nullptr) {
      device.deallocate(t.data());
    }
  }

#ifdef __CUDACC__
  template<typename Scalar>
  __global__ void add_kernel(Scalar * data1, Scalar * data2, long size) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
      data1[idx] += data2[idx];
    }
  }
#endif

  /// <summary>
  ///  Allocate memory on the device for the data
  /// </summary>
  /// <typeparam name="Device_"></typeparam>
  /// <typeparam name="Scalar"></typeparam>
  /// <param name="device"></param>
  /// <param name="dims"></param>
  /// <returns></returns>
  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline TensorView<Scalar, Rank, Layout> create_device_view(const DSizes<Index, Rank>& dims, Device_& device) {

    size_t alloc_size = dims.TotalSize() * sizeof(Scalar);
    Scalar* ptr = static_cast<Scalar*>(device.allocate(alloc_size));
    TensorView<Scalar, Rank, Layout> out(ptr, dims);
    return out;
  }

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline void move_to(TensorView<Scalar, Rank, Layout>& dest, const TensorView<Scalar, Rank, Layout>& src, Device_& device) {

    assert(src.dimensions() == dest.dimensions());
    device.memcpyHostToDevice(dest.data(), src.data(), dest.dimensions().TotalSize() * sizeof(Scalar));
  }

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline void move_to(TensorView<Scalar, Rank, Layout>& dest, const Scalar* src, Device_& device) {

    device.memcpyHostToDevice(dest.data(), src, dest.dimensions().TotalSize() * sizeof(Scalar));
  }

  /// <summary>
  /// Just line t.setConstant but extending for GPU
  /// Assuming memory has been allocated
  /// </summary>
  /// <typeparam name="Device_"></typeparam>
  /// <typeparam name="Scalar"></typeparam>
  /// <param name="t"></param>
  /// <param name="val"></param>
  /// <param name="device"></param>
  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline void setConstant(const TensorView<Scalar, Rank, Layout>& t, Scalar val, Device_& device) {

    assert(t.size());
    std::vector<Scalar> const_mem(t.size());
    std::fill(const_mem.begin(), const_mem.end(), val);

    size_t final_size = t.size() * sizeof(Scalar);
    device.memcpyHostToDevice(t.data(), const_mem.data(), final_size);
  }


  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline void setZero(const TensorView<Scalar, Rank, Layout>& t, Device_& device) {
    setConstant(t, (Scalar)0, device);
  }

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  inline void setValues(TensorView<Scalar, Rank, Layout>& t, const Tensor<Scalar, Rank, Layout>& vals, Device_& device) {

    assert(t.size());
    size_t final_size = t.size() * sizeof(Scalar);
    device.memcpyHostToDevice(t.data(), vals.data(), final_size);
  }

  /// <summary>
  /// Given a flat index into dimensions, return offsets along each dimension
  /// </summary>
  /// <param name="source_dim">Original dimensions</param>
  /// <param name="idx">Flat index</param>
  /// <returns>Offset along each dimension</returns>
  template<typename IntIndex, int Rank, int Layout = RowMajor>
  EIGEN_DEVICE_FUNC inline DSizes<IntIndex, Rank> from_flat_dim(const DSizes<IntIndex, Rank> source_dim, IntIndex idx) {

    DSizes<IntIndex, Rank> out;
    IntIndex offset = idx;

    if (Layout == ColMajor) {
      for (IntIndex i = 0; i < Rank; i++) {
        out[i] = offset % source_dim[i];
        offset /= source_dim[i];
      }
    }
    else {
      for (IntIndex i = Rank - 1; i >= 0; i--) {
        out[i] = offset % source_dim[i];
        offset /= source_dim[i];
      }
    }
    return out;
  }

  /// <summary>
  /// Given tensor dimensions and an offset, compute the offset flat index
  /// </summary>
  /// <param name="source_dim">Original dimension</param>
  /// <param name="offsets">Offset</param>
  /// <returns>Flat index</returns>
  template<typename IntIndex, int Rank, int Layout = RowMajor>
  EIGEN_DEVICE_FUNC inline IntIndex to_flat_dim(const array<IntIndex, Rank> source_dim, const array<IntIndex, Rank> offsets) {

#ifndef __CUDA_ARCH__
    for (IntIndex i = 0; i < Rank; i++) {
      if (offsets[i] >= source_dim[i] || offsets[i] < 0 || source_dim[i] < 1) {
        throw std::invalid_argument("Wrong range for dimensions and/or offsets: dimension " + std::to_string(i));
      }
    }
#endif

    IntIndex res = 0;
    if (Layout == ColMajor) {
      res = offsets[Rank - 1];
      for (IntIndex i = Rank - 2; i >= 0; i--) {
        res *= source_dim[i];
        res += offsets[i];
      }
    }
    else {
      res = offsets[0];
      for (IntIndex i = 1; i < Rank; i++) {
        res *= source_dim[i];
        res += offsets[i];
      }
    }
    return res;
  }

  template<typename ToIndex, long Rank>
  EIGEN_DEVICE_FUNC inline DSizes<ToIndex, Rank> dimensions_cast(DSizes<Index, Rank> from) {

    DSizes<ToIndex, Rank> out;
    for (long i = 0; i < Rank; i++) {
      out[i] = from[i];
    }
    return out;
  }

  /// <summary>
  /// Given the size of a dimension and a block size
  /// figure out the grid dimension size
  /// </summary>
  /// <param name="total"></param>
  /// <param name="block_size"></param>
  /// <returns></returns>
  inline size_t getGridSize(size_t total, size_t block_size) {
    assert(total > 0 && block_size > 0);
    return (total + block_size - 1) / block_size;
  }

  template<typename Device_>
  EIGEN_DEVICE_FUNC inline bool is_cpu(Device_ device) {
    return std::is_same<Device_, DefaultDevice>::value || std::is_same<Device_, ThreadPoolDevice>::value;
  }
} // EigenSinn
