#pragma once

#include "cudnn_workspace.hpp"

const float one = 1.f;
const float zero = 0.f;
const float minus_one = -1.f;

namespace EigenSinn {

  template <typename Scalar>
  class CudnnActivations {

  public:
    CudnnActivations(cudnnTensorDescriptor_t _tensor_desc, cudnnTensorDescriptor_t _dtensor_desc, cudnnActivationMode_t _act_mode, float _relu_coeff =  0.f)
    : act_mode(_act_mode)
    , cudnn_handle(CudnnWorkspace::cudnn())
    , relu_coeff(_relu_coeff)
    , tensor_desc(_tensor_desc) 
    , dtensor_desc(_dtensor_desc) {
      
      checkCudnnErrors(cudnnCreateActivationDescriptor(&act_desc));
      checkCudnnErrors(cudnnCreateActivationDescriptor(&act_desc));
      checkCudnnErrors(cudnnSetActivationDescriptor(act_desc, act_mode, CUDNN_PROPAGATE_NAN, relu_coeff));

    }

    virtual ~CudnnActivations() {
      if (act_desc != nullptr) {
        cudnnDestroyActivationDescriptor(act_desc);
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
        x,
        tensor_data,
        &beta,
        tensor_descriptor,
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
      checkCudnnErrors(cudnnActivationForward(cudnn_handle, 
        act_desc,
        &alpha,
        tensor_desc,
        layer_output,
        dtensor_desc,
        dy,
        tensor_desc,
        layer_output,
        tensor_desc,
        dx
        );
    }

  private:
    cudnnActivationDescriptor_t act_desc = nullptr;
    cudnnHandle_t cudnn_handle;
    cudnnActivationMode_t act_mode;
    cudnnTensorDescriptor_t tensor_desc;
    cudnnTensorDescriptor_t dtensor_desc;
    float relu_coeff;

    float* prev_layer, * layer_output;
    const float alpha = 1.0f, beta = 0.0f;
  };
} // namespace EigenSinn