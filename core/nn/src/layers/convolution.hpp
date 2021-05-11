#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include "ops/initializations.hpp"
#include "helpers/conv_params_bag.hpp"

#ifdef EIGEN_USE_GPU
#include "cudnn/cudnn_workspace.hpp"
#endif

namespace EigenSinn {

  // Batch normalization layers can take care of bias
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class Conv2d : public LayerBase<Scalar, Device_> {

  public:

    Conv2d(const array<Index, 4>& kernelDims, const Padding2D& _padding = { 0, 0 }, const int _stride = 1, const int _dilation = 1)
      : kernel(kernelDims)
      , padding(_padding)
      , stride(_stride)
      , dilation(_dilation)
      , bias(kernelDims[0])
      , bias_broadcast({ 0, 0, 0, 0 })
      , loss_by_bias_derivative(kernelDims[0])
      , is_cudnn(false) {

      }


    // TODO: this needs to be implemented for real
    // Also strides and padding should be added
    void init() override {

      // Wrapping the pointer and moving it to the tensor keeping in mind the GPU device
      kernel = generate_xavier<Scalar, 4, Layout, Device_>(kernel.dimensions());
      bias.setZero();
    }

    void init(Tensor<Scalar, 4, Layout>& _weights) {
      init();

      kernel.set_from_host(_weights);
    }

    void init(Tensor<Scalar, 4, Layout>& _weights, Tensor<Scalar, 1, Layout>& _bias) {
      init(_weights);

      bias.set_from_host(_bias);
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) override {

      DeviceTensor<Scalar, 4, Device_, Layout> prev_layer(prev_layer_any.get_output());

      if(!params || params->check(prev_layer.dimensions())) {
        params = std::make_shared<ConvolutionParams<4>>(prev_layer.dimensions(), kernel.dimensions(), padding, stride, dilation, false, is_cudnn);

        // make sure all output tensors are initialized to right sizes
        dX.resize(params->get_input_dims());
        dW.resize(kernel.dimensions());
        layer_output.resize(params->get_out_dims());
      }

#ifdef EIGEN_USE_GPU
      if (is_cudnn && !cudnn_workspace) {
        cudnn_workspace = std::make_shared<CudnnWorkspace>(*params);
      }


      if (is_cudnn) {
        checkCudnnErrors(cudnnConvolutionForward(CudnnWorkspace::cudnn(), &(CudnnWorkspace::one), cudnn_workspace->input_desc, prev_layer->data(),
          cudnn_workspace->filter_desc, kernel->data(), cudnn_workspace->conv_desc, cudnn_workspace->conv_fwd_algo, cudnn_workspace->d_workspace, cudnn_workspace->workspace_size,
          &(CudnnWorkspace::zero), cudnn_workspace->output_desc, layer_output->data()));

      }
      else {
#endif
        layer_output = convolve<Scalar, 4, Layout, Device_>(prev_layer, kernel, *params);
#ifdef EIGEN_USE_GPU
      }
#endif

      //add bias to each channel
      auto dims = layer_output.dimensions();
      bias_broadcast = { dims[0], 1, dims[2], dims[3] };

      // one bias per filter
      layer_output.view() += bias->reshape(array<Index, 4>{ 1, kernel.dimension(0), 1, 1 }).broadcast(bias_broadcast);

    }

    void backward(LayerBase<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, 4, Device_, Layout> prev_layer(prev_layer_any.get_output());
      DeviceTensor<Scalar, 4, Device_, Layout> next_layer_grad(next_layer_grad_any);

      //bias
      loss_by_bias_derivative.view() = next_layer_grad->sum(array<Index, 3>{0, 2, 3});

#ifdef EIGEN_USE_GPU
      if (is_cudnn) {
        // data backwards
        checkCudnnErrors(cudnnConvolutionBackwardData(CudnnWorkspace::cudnn(), &(CudnnWorkspace::one), cudnn_workspace->filter_desc, kernel->data(),
          cudnn_workspace->output_desc, next_layer_grad->data(), cudnn_workspace->conv_desc, cudnn_workspace->conv_bwd_data_algo,
          cudnn_workspace->d_workspace, cudnn_workspace->workspace_size, &(CudnnWorkspace::zero), cudnn_workspace->input_desc, dX->data()));

        // weights backwards
        checkCudnnErrors(
          cudnnConvolutionBackwardFilter(CudnnWorkspace::cudnn(), &(CudnnWorkspace::one), cudnn_workspace->input_desc, prev_layer->data(),
            cudnn_workspace->output_desc, next_layer_grad->data(),
            cudnn_workspace->conv_desc, cudnn_workspace->conv_bwd_filter_algo, cudnn_workspace->d_workspace,
            cudnn_workspace->workspace_size, &(CudnnWorkspace::zero), cudnn_workspace->filter_desc, dW->data()));

        return;
      }
#endif // EIGEN_USE_GPU

      DeviceTensor<Scalar, 2, Device_, Layout> dout = unfold_conv_res(next_layer_grad);

      // flatten weights and kernel
      DeviceTensor<Scalar, 2, Device_, Layout> unf_kernel = unfold_kernel(kernel);
      DeviceTensor<Scalar, 2, Device_, Layout> x_col = im2col(prev_layer, *params);

      // dX: kernel.T * dout
      DeviceTensor<Scalar, 4, Device_, Layout> dilated = dilate_tensor(kernel, dilation);
      DeviceTensor<Scalar, 2, Device_, Layout> unf_dilated = unfold_kernel(dilated);
      ProductDims prod_dims = { IndexPair<int>(0, 0) };
      DeviceTensor<Scalar, 2, Device_, Layout>  dX_col(unf_dilated.dimension(1), dout.dimension(1));
      dX_col.view() = unf_dilated->contract(*dout, prod_dims);

      // dW: dout * x_col.T
      prod_dims = { IndexPair<int>(1, 1) };
      DeviceTensor<Scalar, 2, Device_, Layout>  dW_col(dout.dimension(0), x_col.dimension(0));
      dW_col.view() = dout->contract(*x_col, prod_dims);

      col2im(dX_col, dX, *params, true);
      dW = fold_kernel(dW_col, kernel.dimensions());
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return dX.raw();
    }

    // feed to optimizer
    PtrTensorAdapter<Scalar, Device_> get_loss_by_weights_derivative() override {
      return dW.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_weights() override {
      return kernel.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_bias() override {
      return bias.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_bias_derivative() override {
      return loss_by_bias_derivative.raw();
    }

    void set_weights(PtrTensorAdapter<Scalar, Device_>& v) override {
      kernel = DeviceTensor<Scalar, 4, Device_, Layout>(v);
    }

    void set_bias(PtrTensorAdapter<Scalar, Device_>& v) override {
      bias = DeviceTensor<Scalar, 1, Device_, Layout>(v);
    }

    inline void set_cudnn(bool _is_cudnn) { 
      assert(!_is_cudnn || Layout & RowMajor);
      is_cudnn = _is_cudnn; 
    }

  private:
    DeviceTensor<Scalar, 4, Device_, Layout> kernel, layer_output, dX, dW;
    DeviceTensor<Scalar, 1, Device_, Layout> bias, loss_by_bias_derivative;

    const int stride, dilation;
    const Padding2D padding;
    array<Index, 4> bias_broadcast;

    // we don't know the input dimension offhand, so default initialization
    std::shared_ptr<ConvolutionParams<4>> params;
#ifdef EIGEN_USE_GPU
    std::shared_ptr<CudnnWorkspace> cudnn_workspace;
#endif
    bool is_cudnn;
  };
}