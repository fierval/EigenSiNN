#pragma once

#include "cudnn_workspace.hpp"
#include "cudnn_tensor_desc.hpp"

namespace EigenSinn {

  template <typename Scalar, int Rank>
  class CudnnPooling {

  public:
    CudnnPooling(DSizes<Index, Rank> in_dims, DSizes<Index, Rank> out_dims, cudnnPoolingMode_t _pooling_mode, MaxPoolParams& params)
    : cudnn_handle(CudnnWorkspace::cudnn())
    , pooling_mode(_pooling_mode)
    , in_desc(in_dims)
    , out_desc(out_dims) {
      
      if (Rank < 4) { return; }
      checkCudnnErrors(cudnnCreatePoolingDescriptor(&pooling_desc));
      checkCudnnErrors(cudnnSetPooling2dDescriptor(pooling_desc, pooling_mode, params.kernel_dims[2], params.kernel_dims[3], params.padding.first, params.padding.second, params.stride, params.stride));

    }

    virtual ~CudnnPooling() {
      if (pooling_desc != nullptr) {
        cudnnDestroyPoolingDescriptor(pooling_desc);
      }
    }

    /// <summary>
    /// assume we get the same kind of tensor after activation pass
    /// hence same descriptor for input and output
    /// </summary>
    /// <param name="x">Input: Tensor prior to activation</param>
    /// <param name="y">Output: Activation result</param>
    void forward(Scalar* x, Scalar * y) {

      checkCudnnErrors(cudnnActivationForward(cudnn_handle,
        act_desc,
        &alpha,
        tensor_desc,
        x,
        &beta,
        tensor_desc,
        y));

      prev_layer = x;
      layer_output = y;
    }

    /// <summary>
    /// assume we get the same kind of tensor as the output of the convolution op
    /// after backward pass through the activation function
    /// </summary>
    /// <param name="dy"></param>
    /// <param name="dx"></param>
    void backward(Scalar* dy, Scalar* dx) {

      checkCudnnErrors(cudnnActivationBackward(cudnn_handle, 
        act_desc,
        &alpha,
        tensor_desc,
        layer_output,
        tensor_desc,
        dy,
        tensor_desc,
        layer_output,
        &beta,
        tensor_desc,
        dx
        ));
    }

  private:
    cudnnPoolingDescriptor_t  pooling_desc = nullptr;
    cudnnPoolingMode_t pooling_mode;
    cudnnHandle_t cudnn_handle;

    TensorDescWrapper<Rank> in_desc;
    TensorDescWrapper<Rank> out_desc;

    float relu_coeff;

    float* prev_layer, * layer_output;
    const float alpha = 1.0f, beta = 0.0f;
  };
} // namespace EigenSinn