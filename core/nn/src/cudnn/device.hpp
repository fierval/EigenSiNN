#pragma once
#include "common.hpp"

#include "ops/opsbase.hpp"

namespace EigenSinn {
// container for cuda resources
  struct CudnnDevice
  {
    CudnnDevice()
    {
      checkCudnnErrors(cudnnCreate(&_cudnn_handle));
    }
    ~CudnnDevice()
    {
      checkCudnnErrors(cudnnDestroy(_cudnn_handle));
    }

    cudnnHandle_t operator()() { return _cudnn_handle; }

  private:
    cudnnHandle_t  _cudnn_handle;
  };


  cudnnTensorDescriptor_t tensor4d(const DSizes<Index, 4>& dims)
  {
    cudnnTensorDescriptor_t tensor_desc;

    cudnnCreateTensorDescriptor(&tensor_desc);
    cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3]);

    return tensor_desc;
  }

  DSizes<Index, 4> set_output_dims(cudnnConvolutionDescriptor_t conv_desc, cudnnTensorDescriptor_t input_desc, cudnnFilterDescriptor_t filter_desc) {

    int dims[4];
    DSizes<Index, 4> out;

    checkCudnnErrors(
      cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &dims[0], &dims[1], &dims[2], &dims[3]));

    for (int i = 0; i < 4; i++) {
      out[i] = static_cast<Index>(dims[i]);
    }
    return out;
  }

} // namespace EigenSinn