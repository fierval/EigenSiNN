#pragma once
#include "ops/opsbase.hpp"

using namespace Eigen;
using std::unique_ptr;
using std::make_unique;

namespace EigenSinn {

  template<typename Device_, typename Scalar, Index Rank>
  inline void free(TensorView<Scalar, Rank>& t, Device_& device) {

    if (t.data() != nullptr) {
      device.deallocate(t.data());
    }

  }

  /// <summary>
  ///  Allocate memory on the device for the data
  /// </summary>
  /// <typeparam name="Device_"></typeparam>
  /// <typeparam name="Scalar"></typeparam>
  /// <param name="device"></param>
  /// <param name="dims"></param>
  /// <returns></returns>
  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline TensorView<Scalar, Rank, Layout> create_device_view(const DSizes<Index, Rank>& dims, Device_& device)  {

    size_t alloc_size = dims.TotalSize() * sizeof(Scalar);
    Scalar* ptr = static_cast<Scalar*>(device.allocate(alloc_size));
    TensorView<Scalar, Rank, Layout> out(ptr, dims);
    return out;
  }

  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline void move_to(TensorView<Scalar, Rank, Layout>& dest, const TensorView<Scalar, Rank, Layout>& src, Device_& device) {

    assert(src.dimensions() == dest.dimensions());
    device.memcpyHostToDevice(dest.data(), src.data(), dest.dimensions().TotalSize() * sizeof(Scalar));
  }

  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline void move_to(TensorView<Scalar, Rank, Layout>& dest, const Scalar * src, Device_& device) {

    device.memcpyHostToDevice(dest.data(), src, dest.dimensions().TotalSize() * sizeof(Scalar));
  }

  /// <summary>
  /// Allocate original tensor
  /// </summary>
  /// <typeparam name="Device_"></typeparam>
  /// <typeparam name="Scalar"></typeparam>
  /// <param name="dims"></param>
  /// <param name="device"></param>
  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline PtrTensorView<Scalar, Rank> create_device_ptr(DSizes<Index, Rank> dims, Device_& device) {

    size_t alloc_size = dims.TotalSize() * sizeof(Scalar);
    Scalar* ptr = static_cast<Scalar*>(device.allocate(alloc_size));
    PtrTensorView<Scalar, Rank, Layout> out(new TensorView<Scalar, Rank, Layout>(ptr, dims));

    return std::move(out);
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
  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline void setConstant(TensorView<Scalar, Rank, Layout>& t, Scalar val, Device_& device) {

    assert(t.size());
    std::vector<Scalar> const_mem(t.size());
    std::fill(const_mem.begin(), const_mem.end(), val);

    size_t final_size = t.size() * sizeof(Scalar);
    device.memcpyHostToDevice(t.data(), const_mem.data(), final_size);
  }


  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline void setZero(TensorView<Scalar, Rank, Layout>& t, Device_& device) {
    setConstant(t, (Scalar)0, device);
  }

  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline void setValues(TensorView<Scalar, Rank, Layout>& t, const Tensor<Scalar, Rank, Layout>& vals, Device_& device) {

    assert(t.size());
    size_t final_size = t.size() * sizeof(Scalar);
    device.memcpyHostToDevice(t.data(), vals.data(), final_size);
  }

  /// <summary>
  /// Given tensor dimensions and an offset, compute the offset flat index
  /// </summary>
  /// <param name="source_dim">Original dimension</param>
  /// <param name="offsets">Offset</param>
  /// <returns>Flat index</returns>
  template<Index Rank, int Layout = ColMajor>
  EIGEN_DEVICE_FUNC inline Index to_flat_dim(const array<Index, Rank> source_dim, const array<Index, Rank> offsets) {

    for (Index i = 0; i < Rank; i++) {
      if (offsets[i] >= source_dim[i] || offsets[i] < 0 || source_dim[i] < 1) {
        throw std::invalid_argument("Wrong range for dimensions and/or offsets: dimension " + std::to_string(i));
      }
    }

    Index res = 0;
    if (Layout == ColMajor) {
      res = offsets[Rank - 1];
      for (Index i = Rank - 2; i >= 0; i--) {
        res *= source_dim[i];
        res += offsets[i];
      }
    }
    else {
      res = offsets[0];
      for (Index i = 1; i < Rank; i++) {
        res *= source_dim[i];
        res += offsets[i];
      }
    }
    return res;
  }
}

/// <summary>
/// Given a flat index into dimensions, return offsets along each dimension
/// </summary>
/// <param name="source_dim">Original dimensions</param>
/// <param name="idx">Flat index</param>
/// <returns>Offset along each dimension</returns>
template<Index Rank, int Layout = ColMajor>
EIGEN_DEVICE_FUNC inline array<Index, Rank> from_flat_dim(const array<Index, Rank> source_dim, Index idx) {

  for (Index i = 0, p = 1; i < Rank; i++) {
    if (((p *= source_dim[i]) > idx && i != Rank - 1) || (i == Rank - 1 && p <= idx)) {
      throw std::invalid_argument("Wrong offset into the source dimension at dimension " + std::to_string(i));
    }
  }

  array<Index, Rank> out;
  Index offset = idx;

  if (Layout == ColMajor) {
    for (Index i = 0; i < Rank; i++) {
      out[i] = offset % source_dim[i];
      offset /= source_dim[i];
    }
  }
  else {
    for (Index i = Rank - 1; i >= 0; i--) {
      out[i] = offset % source_dim[i];
      offset /= source_dim[i];
    }
  }
  return out;
}
