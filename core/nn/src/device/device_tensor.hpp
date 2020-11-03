#pragma once
#include "ops/opsbase.hpp"

#ifdef EIGEN_USE_GPU
#include <cudnn.h>
#include <vector>
#include <algorithm>
#include "helper.h"
#endif

using namespace Eigen;
using std::unique_ptr;
using std::make_unique;

namespace EigenSinn {

#ifdef EIGEN_USE_GPU
  template<typename Scalar, int Dims, int Layout, int Outlayout = Layout>
  inline Scalar* to_device(Tensor<Scalar, Dims, Layout>& t) {

    Scalar* tensor_data = t.data();
    Scalar* dtensor_data;
    array<Index, Dims> dims = t.dimensions();

    bool convert_to_row_major = !(Layout & Eigen::RowMajorBit) && (Outlayout & Eigen::RowMajorBit);

    if (convert_to_row_major && Layout == Eigen::ColMajor) {
      Tensor<Scalar, Dims, Eigen::RowMajor> t_row(t.dimensions());

      // reverse the dimensions
      array<Index, Dims> rev = reverse_dims(t.dimensions());

      t_row = t.swap_layout().shuffle(rev);
      tensor_data = t.data();
    }

    cudaMalloc((void**)&dtensor_data, sizeof(Scalar) * t.size());
    cudaCheckError();

    cudaMemcpy(dtensor_data, tensor_data, sizeof(Scalar) * t.size(), cudaMemcpyHostToDevice);
    cudaCheckError();

    return dtensor_data;
  }

  template<typename Scalar, int Dims, int Layout, int Outlayout = Layout>
  inline TensorMap<Tensor<Scalar, Dims>>* to_gpu_tensor(Tensor<Scalar, Dims, Layout>& t) {

    Scalar* t_data = to_device(t);
    TensorMap<Tensor<Scalar, Dims>>* t_map = new TensorMap<Tensor<Scalar, Dims>>(t_data, t.dimensions());
    return t_map;
  }

  template<typename Scalar, int Dims, int Layout = ColMajor>
  inline Tensor<Scalar, Dims, Layout> from_device(Scalar* dt, array<Index, Dims> dims) {

    size_t size = 1;
    for (Index i = 0; i < Dims; i++) { size *= dims[i]; }

    Scalar* data = (Scalar*)std::malloc(sizeof(Scalar) * size);
    cudaMemcpy((void*)data, dt, sizeof(Scalar) * size, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(dt);
    TensorMap<Tensor<Scalar, Dims, Layout>> out_map(data, dims);
    Tensor<Scalar, Dims, Layout> out(dims);

    out = out_map;
    free(data);
    return out;
  }

  template<typename Scalar>
  inline Tensor<Scalar, 0> from_device(Scalar* dt) {
    return from_device(dt, array<Index, 0>{});
  }

  template<typename Scalar, int Dims, int Layout = ColMajor>
  inline Tensor<Scalar, Dims> from_gpu_tensor(TensorMap<Tensor<Scalar, Dims, Layout>>* t_map) {

    return from_device(t_map->data(), t_map->dimensions());
  }
#endif

  template<typename Device_, typename Scalar, Index Rank>
  inline void free(TensorView<Scalar, Rank>& t, Device_& device) {

    if (t && t->data() != nullptr) {
      device.deallocate(t->data());
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
  inline void move_to(TensorView<Scalar, Rank, Layout>& dest, TensorView<Scalar, Rank, Layout>& src, Device_& device) {

    assert(src.dimensions() == dest.dimensions());
    device.memcpyHostToDevice(dest.data(), src.data(), dest.dimensions().TotalSize() * sizeof(Scalar));
  }

  /// <summary>
  /// Allocate original tensor
  /// </summary>
  /// <typeparam name="Device_"></typeparam>
  /// <typeparam name="Scalar"></typeparam>
  /// <param name="t"></param>
  /// <param name="dims"></param>
  /// <param name="device"></param>
  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline auto resize(TensorView<Scalar, Rank, Layout>& t, DSizes<Index, Rank> dims, Device_& device) {

    if (t.data() != nullptr) {
      device.deallocate(t.data());
    }
    return create_device_view<Device_, Scalar, Rank, Layout>(dims, device);
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

    assert(t);
    std::vector<Scalar> const_mem(t->size());
    std::fill(const_mem.begin(), const_mem.end(), val);

    size_t final_size = t->size() * sizeof(Scalar);
    device.memcpyHostToDevice(t->data(), const_mem.data(), final_size);
  }


  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  inline void setZero(TensorView<Scalar, Rank, Layout>& t, Device_& device) {
    setConstant(t, 0, device);
  }
}
