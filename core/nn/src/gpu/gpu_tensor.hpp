#pragma once
#ifdef EIGEN_USE_GPU
#include <unsupported/Eigen/CXX11/Tensor>
#include "ops/opsbase.hpp"
#include <cudnn.h>
#include <vector>
#include <algorithm>
#include "helper.h"

using namespace Eigen;

namespace EigenSinn {

  template<typename Scalar, int Dims, int Layout, int Outlayout = Layout>
  inline Scalar * to_device(Tensor<Scalar, Dims, Layout>& t) {

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
  inline Tensor<Scalar, Dims, Layout> from_device(Scalar * dt, array<Index, Dims> dims) {

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

  template<typename Scalar>
  inline void free_gpu(Scalar * d_data) {

    cudaFree(d_data);
  }

  template<typename Scalar, int Dims, int Layout = ColMajor>
  inline Tensor<Scalar, Dims> from_gpu_tensor(TensorMap<Tensor<Scalar, Dims, Layout>>* t_map) {

    return from_device(t_map->data(), t_map->dimensions());
  }
}
#endif