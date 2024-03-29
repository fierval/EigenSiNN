#pragma once

#include "helpers/conv_params_bag.hpp"
#include "cudnn_tensor_desc.hpp"

namespace EigenSinn {

  struct CudnnConvWorkspaceWrapper {
    CudnnConvWorkspaceWrapper(size_t size)  
    : workspace_size(size) {

      checkCudaErrors(cudaMalloc((void**)&d_workspace, workspace_size));
    }

    ~CudnnConvWorkspaceWrapper() {
      if (d_workspace != nullptr) { cudaFree(d_workspace); }
    }

    operator void* () { return d_workspace; }

    void* d_workspace = nullptr;
    size_t workspace_size;

  };

  struct CudnnWorkspace {

    CudnnWorkspace(ConvolutionParams<4>& params)
      : input_desc(params.orig_dims())
      , output_desc(params.output_dims()) {

      DSizes<Index, 4> kernel_dims = params.kernel_dims;
      int pad_h = static_cast<int>(params.padding.first);
      int pad_w = static_cast<int>(params.padding.second);
      int stride = params.stride, dilation = params.dilation;

      // Kernel properties
      cudnnCreateFilterDescriptor(&filter_desc);
      checkCudnnErrors(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernel_dims[0], kernel_dims[1], kernel_dims[2], kernel_dims[3]));

      // Convolution descriptor, set properties
      cudnnCreateConvolutionDescriptor(&conv_desc);
      checkCudnnErrors(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

      // setting cudnn convolution math type
      // CUDNN_DEFAULT_MATH operates convolution with FP32.
      // If you use A100, CUDNN utilise tensor cores with TF32.
      checkCudnnErrors(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

      set_workspace();

    }

    ~CudnnWorkspace() {
      cudnnDestroyFilterDescriptor(filter_desc);
      cudnnDestroyConvolutionDescriptor(conv_desc);
    }

    // weight/bias descriptor
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t bias_desc;

    cudnnConvolutionDescriptor_t    conv_desc;

    cudnnConvolutionFwdAlgo_t       conv_fwd_algo;
    cudnnConvolutionBwdDataAlgo_t   conv_bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo;
    TensorDescWrapper<4>         input_desc;
    TensorDescWrapper<4>         output_desc;

    static inline std::mutex workspace_mutex;
    static inline size_t workspace_size = 0;
    static inline std::unique_ptr<CudnnConvWorkspaceWrapper> d_workspace;

    static inline float one = 1.f;
    static inline float zero = 0.f;
    static inline float minus_one = -1.f;
    static inline cudnnHandle_t cudnn_handle;

    inline static cudnnHandle_t cudnn() {
      static std::once_flag onceFlag;

      std::call_once(onceFlag, []() {checkCudnnErrors(cudnnCreate(&cudnn_handle)); });
      return cudnn_handle;
    }

    inline static void * workspace() {
      if (!d_workspace) {
        return nullptr;
      }
      return d_workspace->d_workspace;
    }

  private:
    inline void set_workspace()
    {

      // forward

      int algo_max_count;
      int returnedAlgoCount = 0;

      // fwd algorithm
      checkCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn(), &algo_max_count));
      std::vector<cudnnConvolutionFwdAlgoPerf_t> 		 fwd_algoperf_results(algo_max_count);

      checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(cudnn(),
        input_desc, filter_desc, conv_desc, output_desc,
        algo_max_count, &returnedAlgoCount, &fwd_algoperf_results[0]));

      conv_fwd_algo = fwd_algoperf_results[0].algo;

      // bwd - filter
      checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn(), &algo_max_count));
      std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algoperf_results(algo_max_count);

      checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn(),
        input_desc, output_desc, conv_desc, filter_desc,
        algo_max_count, &returnedAlgoCount, &bwd_filter_algoperf_results[0]));

      conv_bwd_filter_algo = bwd_filter_algoperf_results[0].algo;

      // bwd - data
      checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn(), &algo_max_count));
      std::vector<cudnnConvolutionBwdDataAlgoPerf_t>	 bwd_data_algoperf_results(algo_max_count);

      checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn(),
        filter_desc, output_desc, conv_desc, input_desc,
        algo_max_count, &returnedAlgoCount, &bwd_data_algoperf_results[0]));

      conv_bwd_data_algo = bwd_data_algoperf_results[0].algo;

      // workspace
      // workspace is shared between all convolutional layers
      std::lock_guard<std::mutex> lock(workspace_mutex);
      size_t cur_workspace_size = 0;
      size_t temp_size = 0;

      checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(cudnn(),
        input_desc, filter_desc, conv_desc, output_desc,
        conv_fwd_algo, &temp_size));

#undef max
      // There is an Eigen::max constant set to 0 which causes the following line(s) to be ignored.
      // Should therefore specify the global namespace explicitly!!!
      cur_workspace_size = std::max(cur_workspace_size, temp_size);

      checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn(),
        input_desc, output_desc, conv_desc, filter_desc,
        conv_bwd_filter_algo, &temp_size));
      cur_workspace_size = std::max(cur_workspace_size, temp_size);

      checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn(),
        filter_desc, output_desc, conv_desc, input_desc,
        conv_bwd_data_algo, &temp_size));

      cur_workspace_size = std::max(cur_workspace_size, temp_size);
      
      if (cur_workspace_size > workspace_size)
      {
        workspace_size = cur_workspace_size;
        d_workspace.reset(new CudnnConvWorkspaceWrapper(workspace_size));
      }

    }

  };
} // namespace EigenSinn