#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <cudnn.h>

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
      array<Index, Dims> rev = t.dimensions();
      for (Index i = rev.size() - 1; i > 0; i--) {
        rev[rev.size() - i - 1] = dims[i];
      }
      dims = rev;
      t_row = t. swap_layout().shuffle(dims);
      tensor_data = t.data();
    }

    cudaMalloc((void**)&dtensor_data, sizeof(Scalar) * t.size());
    cudaCheckError();

    cudaMemcpy(dtensor_data, tensor_data, sizeof(Scalar) * t.size(), cudaMemcpyHostToDevice);
    cudaCheckError();

    return dtensor_data;
  }

  template<typename Scalar>
  inline Scalar * from_gpu(Scalar * dt, size_t size) {

    Scalar* data = (Scalar*)std::malloc(sizeof(Scalar) * size);
    cudaMemcpy((void*)data, dt, sizeof(Scalar) * size, cudaMemcpyDeviceToHost);
    cudaCheckError();

    return data;
  }

  template<typename Scalar>
  inline void free_gpu(Scalar * d_data) {

    cudaFree(d_data);
  }
}