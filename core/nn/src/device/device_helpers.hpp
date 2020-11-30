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

  template<Index Rank, int Layout = ColMajor>
  inline Index compute_flat_dim(const array<Index, Rank> source_dim, const array<Index, Rank> offsets) {

    Index res = 0;
    if (Layout == ColMajor) {
      res = source_dim[Rank - 2] * offsets[Rank - 1];
      for (int i = Rank - 2; i >= 0; i--) {
        res += offsets[i + 1];
        res *= source_dim[i];
      }
      res += offsets[0];
    }
    else {
      res = source_dim[1] * offsets[0];
      for (int i = 2; i < Rank; i++) {
        res += offsets[i - 1];
        res *= source_dim[i];
      }
      res += offsets[Rank];
    }
    return res;
  }
}
