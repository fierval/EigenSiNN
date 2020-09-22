#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include "ops/opsbase.hpp"
#include <cudnn.h>
#include <vector>
#include <algorithm>

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#endif
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

using namespace Eigen;

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   throw std::runtime_error(cudaGetErrorString(e)); \
 }                                                                 \
}

namespace EigenSinn {

  template<typename Scalar, int Dims, int Layout, int Outlayout = Layout>
  inline auto to_gpu(Tensor<Scalar, Dims, Layout>& t) {

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

  template<typename Scalar, int Dims>
  inline Tensor<Scalar, Dims> from_gpu(Scalar * dt, array<Index, Dims> dims) {

    size_t size = 1;
    for (Index i = 0; i < Dims; i++) { size *= dims[i]; }

    Scalar* data = (Scalar*)std::malloc(sizeof(Scalar) * size);
    cudaMemcpy((void*)data, dt, sizeof(Scalar) * size, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(dt);
    TensorMap<Tensor<Scalar, Dims, RowMajor>> out_map(data, dims);
    Tensor<Scalar, Dims> out(dims);

    array<Index, Dims> rev_dims = reverse_dims(dims);
    out = out_map.swap_layout().shuffle(rev_dims);
    return out;
  }

  template<typename Scalar>
  inline void free_gpu(Scalar * d_data) {

    cudaFree(d_data);
  }
}