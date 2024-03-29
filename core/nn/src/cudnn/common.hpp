#pragma once

#include <cudnn.h>
#include <cublas_v2.h>

#include <curand.h>

namespace EigenSinn
{

  /* CUDA API error return checker */
#ifndef checkCudaErrors
#define checkCudaErrors(err)                                                                        \
    {                                                                                               \
        if (err != cudaSuccess)                                                                     \
        {                                                                                           \
            fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, cudaGetErrorString(err), __FILE__, __LINE__);                              \
      fprintf(stderr, "%d\n", cudaSuccess);													\
            exit(-1);                                                                               \
        }                                                                                           \
    }
#endif

#define checkCudnnErrors(err)                                                                        \
    {                                                                                               \
        if (err != CUDNN_STATUS_SUCCESS)                                                                     \
        {                                                                                           \
            fprintf(stderr, "checkCudnnErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, cudnnGetErrorString(err), __FILE__, __LINE__);                              \
      fprintf(stderr, "%d\n", cudaSuccess);													\
            exit(-1);                                                                               \
        }                                                                                           \
    }

  static const char* _cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
  }

#define checkCublasErrors(err)                                                                        \
    {                                                                                                 \
        if (err != CUBLAS_STATUS_SUCCESS)                                                             \
        {                                                                                             \
            fprintf(stderr, "checkCublasErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, _cublasGetErrorEnum(err), __FILE__, __LINE__);                                 \
            exit(-1);                                                                                 \
        }                                                                                             \
    }

  // cuRAND API errors
  static const char* _curandGetErrorEnum(curandStatus_t error) {
    switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
  }


#define checkCurandErrors(err)                                                                        \
    {                                                                                                \
        if (err != CURAND_STATUS_SUCCESS)                                                             \
        {                                                                                            \
            fprintf(stderr, "checkCurandErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, _curandGetErrorEnum(err), __FILE__, __LINE__);                              \
            exit(-1);                                                                                \
        }                                                                                            \
    }
}

// common functions
inline cudnnTensorDescriptor_t tensor4d(const DSizes<Index, 4>& dims)
{
  cudnnTensorDescriptor_t tensor_desc;

  cudnnCreateTensorDescriptor(&tensor_desc);
  cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3]);

  return tensor_desc;
}

inline DSizes<Index, 4> set_output_dims(cudnnConvolutionDescriptor_t conv_desc, cudnnTensorDescriptor_t input_desc, cudnnFilterDescriptor_t filter_desc) {

  int dims[4];
  DSizes<Index, 4> out;

  checkCudnnErrors(
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &dims[0], &dims[1], &dims[2], &dims[3]));

  for (int i = 0; i < 4; i++) {
    out[i] = static_cast<Index>(dims[i]);
  }
  return out;
}


