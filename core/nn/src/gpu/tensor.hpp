#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <cuda_runtime.h>
#include <cudnn.h>


using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::array;
using Eigen::Index;
using std::begin;
using std::end;

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   throw(-1); \
 }                                                                 \
}

namespace EigenSinn {

  template<typename Scalar, Index Dims, int Layout>
  inline auto to_gpu(const Tensor<Scalar, Dims, Layout>& t, bool convert_to_row_major = false) {

    Scalar* tensor_data = t.data();
    Scalar* dtensor_data;
    array<Index, Dims> dims = t.dimensions();

    if (convert_to_row_major && Layout == Eigen::ColMajor) {
      Tensor<Scalar, Dims, Eigen::RowMajor> t_row(t.dimensions());

      std::reverse(begin(dims), end(dims));
      t_row = t. swap_layout().shuffle(dims);
      tensor_data = t.data();
    }

    cudaMalloc((void**)&dtensor_data, sizeof(Scalar) * t.size());
    cudaCheckError();

    cudaMemcpy(dtensor_data, tensor_data, sizeof(Scalar) * t.size(), cudaMemcpyHostToDevice);
    cudaCheckError();

    const int layout = convert_to_row_major ? Eigen::RowMajor : Layout;
    TensorMap<Tensor<Scalar, Dims, layout>> gpu_tensor(dtensor_data, dims);
    return gpu_tensor;
  }
}