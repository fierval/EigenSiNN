#pragma once

#include <cudnn.h>
#include <cublas_v2.h>

#include <curand.h>

namespace EigenSinn
{
#define BLOCK_DIM_1D    512
#define BLOCK_DIM       16

  /* DEBUG FLAGS */
#define DEBUG_FORWARD   0
#define DEBUG_BACKWARD  0

#define DEBUG_CONV      0
#define DEBUG_DENSE     0
#define DEBUG_SOFTMAX   0
#define DEBUG_UPDATE    0

#define DEBUG_LOSS      0
#define DEBUG_ACCURACY  0

#define DEBUG_FIND_ALGO 1

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

  // container for cuda resources
  class CudaContext
  {
  public:
    CudaContext()
    {
      cublasCreate(&_cublas_handle);
      checkCudaErrors(cudaGetLastError());
      checkCudnnErrors(cudnnCreate(&_cudnn_handle));
    }
    ~CudaContext()
    {
      cublasDestroy(_cublas_handle);
      checkCudnnErrors(cudnnDestroy(_cudnn_handle));
      cudnnDestroyFilterDescriptor(filter_desc_);
      cudnnDestroyConvolutionDescriptor(conv_desc_);

      // terminate internal created blobs
      if (d_workspace_ != nullptr) { cudaFree(d_workspace_);	d_workspace_ = nullptr; }
    }

    cublasHandle_t cublas() {
      //std::cout << "Get cublas request" << std::endl; getchar();
      return _cublas_handle;
    };
    cudnnHandle_t cudnn() { return _cudnn_handle; };

    const float one = 1.f;
    const float zero = 0.f;
    const float minus_one = -1.f;

    void* set_workspace()
    {
      size_t temp_size = 0;

      // forward
      std::vector<cudnnConvolutionFwdAlgoPerf_t> 		 fwd_algo_perf_results(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
      std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algo_perf_results(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
      std::vector<cudnnConvolutionBwdDataAlgoPerf_t>	 bwd_data_algo_perf_results(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);

      int algo_max_count;
      int returnedAlgoCount = 0;

      checkCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
      std::cout << ": Available Algorithm Count [FWD]: " << algo_max_count << std::endl;
      checkCudnnErrors(cudnnFindConvolutionForwardAlgorithm(cudnn(),
        input_desc_, filter_desc_, conv_desc_, output_desc_,
        algo_max_count, &returnedAlgoCount, &fwd_algo_perf_results[0]));
      std::cout << "returned algo_count: " << returnedAlgoCount << std::endl;
      for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << "fwd algo[" << i << "] time: " << fwd_algo_perf_results[i].time << ", memory: " << fwd_algo_perf_results[i].memory << std::endl;
#else
      checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(cudnn(),
        input_desc_, filter_desc_, conv_desc_, output_desc_,
        algo_max_count, &returnedAlgoCount, &fwd_algo_perf_results[0]));
#endif
      // shoose the fastest algorithm
      conv_fwd_algo_ = fwd_algo_perf_results[0].algo;
      checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(cudnn(),
        input_desc_, filter_desc_, conv_desc_, output_desc_,
        conv_fwd_algo_, &temp_size));

      workspace_size_ = max(workspace_size_, temp_size);

      // bwd - filter
      checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
      std::cout << ": Available Algorithm Count [BWD-filter]: " << algo_max_count << std::endl;
      checkCudnnErrors(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn(),
        input_desc_, output_desc_, conv_desc_, filter_desc_,
        algo_max_count, &returnedAlgoCount, &bwd_filter_algo_perf_results[0]));
      for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << "bwd filter algo[" << i << "] time: " << fwd_algo_perf_results[i].time << ", memory: " << fwd_algo_perf_results[i].memory << std::endl;
#else
      checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cuda_->cudnn(),
        input_desc_, output_desc_, conv_desc_, filter_desc_,
        algo_max_count, &returnedAlgoCount, &bwd_filter_algo_perf_results[0]));
#endif
      conv_bwd_filter_algo_ = bwd_filter_algo_perf_results[0].algo;
      checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn(),
        input_desc_, output_desc_, conv_desc_, filter_desc_,
        conv_bwd_filter_algo_, &temp_size));
      workspace_size_ = max(workspace_size_, temp_size);

      // bwd - data
      checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
      std::cout << ": Available Algorithm Count [BWD-data]: " << algo_max_count << std::endl;
      checkCudnnErrors(cudnnFindConvolutionBackwardDataAlgorithm(cudnn(),
        filter_desc_, output_desc_, conv_desc_, input_desc_,
        algo_max_count, &returnedAlgoCount, &bwd_data_algo_perf_results[0]));
      for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << "bwd data algo[" << i << "] time: " << fwd_algo_perf_results[i].time << ", memory: " << fwd_algo_perf_results[i].memory << std::endl;
#else
      checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn(),
        filter_desc_, output_desc_, conv_desc_, input_desc_,
        algo_max_count, &returnedAlgoCount, &bwd_data_algo_perf_results[0]));
#endif
      conv_bwd_data_algo_ = bwd_data_algo_perf_results[0].algo;
      checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn(),
        filter_desc_, output_desc_, conv_desc_, input_desc_,
        conv_bwd_data_algo_, &temp_size));
      workspace_size_ = max(workspace_size_, temp_size);

      if (workspace_size_ > 0)
      {
        if (d_workspace_ != nullptr)
          checkCudaErrors(cudaFree(d_workspace_));
        checkCudaErrors(cudaMalloc((void**)&d_workspace_, workspace_size_));
      }

      return d_workspace_;
    }
  private:


    cublasHandle_t _cublas_handle;
    cudnnHandle_t  _cudnn_handle;

    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    // weight/bias descriptor
    cudnnFilterDescriptor_t filter_desc_;
    cudnnTensorDescriptor_t bias_desc_;

    cudnnConvolutionDescriptor_t    conv_desc_;

    cudnnConvolutionFwdAlgo_t       conv_fwd_algo_;
    cudnnConvolutionBwdDataAlgo_t   conv_bwd_data_algo_;
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_;

    size_t workspace_size_ = 0;
    
    void** d_workspace_ = nullptr;

  };
} // namespace EigenSinn