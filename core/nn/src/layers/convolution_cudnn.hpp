#pragma once

#include "layer_base.hpp"
#include "ops/initializations.hpp"
#include "helpers/conv_params_bag.hpp"
#include "cudnn/device.hpp"
#include "helpers/cudnn_workspace.hpp"
#include "convolution_def.hpp"

namespace EigenSinn {
  // REVIEW: cuDNN is marked by the RowMajor layout of tensors through template specialization. 
  // Kinda not ideal!
  // anything RowMajor and we assume cuDNN device
  template <typename Scalar, typename Device_>
  class Conv2d<Scalar, Device_, RowMajor> : public LayerBase<Scalar, Device_> {

  public:

    Conv2d(const array<Index, 4>& kernelDims, const Padding2D& _padding = { 0, 0 }, const int _stride = 1, const int _dilation = 1)
      : kernel(kernelDims)
      , padding(_padding)
      , stride(_stride)
      , dilation(_dilation)
      , bias(kernelDims[0])
      , bias_broadcast({ 0, 0, 0, 0 })
      , loss_by_bias_derivative(kernelDims[0]) {

    }


    void init() override {

      // Wrapping the pointer and moving it to the tensor keeping in mind the GPU device
      kernel = generate_xavier<Scalar, 4, RowMajor, Device_>(kernel.dimensions());
      bias.setZero();
    }

    void init(Tensor<Scalar, 4, RowMajor>& _weights) {
      init();

      kernel.set_from_host(_weights);
    }

    void init(Tensor<Scalar, 4, RowMajor>& _weights, Tensor<Scalar, 1, RowMajor>& _bias) {
      init(_weights);

      bias.set_from_host(_bias);
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) override {

      DeviceTensor<Scalar, 4, Device_, RowMajor> prev_layer(prev_layer_any.get_output());

      if (!params || params->check(prev_layer.dimensions())) {
        params = std::make_shared<ConvolutionParams<4>>(prev_layer.dimensions(), kernel.dimensions(), padding, stride, dilation, false);
        dX.resize(params->orig_dims());
        dW.resize(kernel.dimensions());
      }

      if (!cudnn_workspace) {
        cudnn_workspace = std::make_shared<CudnnWorkspace<Device_>>(prev_layer.device(), *params);
      }

      layer_output.resize(params->output_dims());

      checkCudnnErrors(cudnnConvolutionForward((*cudnn_workspace)(), &(cudnn_workspace->one), cudnn_workspace->input_desc, prev_layer->data(),
        cudnn_workspace->filter_desc, kernel->data(), cudnn_workspace->conv_desc, cudnn_workspace->conv_fwd_algo, cudnn_workspace->d_workspace, cudnn_workspace->workspace_size,
        &(cudnn_workspace->zero), cudnn_workspace->output_desc, layer_output->data()));

      //add bias to each channel
      auto dims = layer_output.dimensions();
      bias_broadcast = { dims[0], 1, dims[2], dims[3] };

      // one bias per filter
      layer_output.view() += bias->reshape(array<Index, 4>{ 1, kernel.dimension(0), 1, 1 }).broadcast(bias_broadcast);

    }

    void backward(LayerBase<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, 4, Device_, RowMajor> prev_layer(prev_layer_any.get_output());
      DeviceTensor<Scalar, 4, Device_, RowMajor> next_layer_grad(next_layer_grad_any);

      //bias
      loss_by_bias_derivative.view() = next_layer_grad->sum(array<Index, 3>{0, 2, 3});

      // data backwards
      checkCudnnErrors(cudnnConvolutionBackwardData((*cudnn_workspace)(), &(cudnn_workspace->one), cudnn_workspace->filter_desc, kernel->data(),
        cudnn_workspace->output_desc, next_layer_grad->data(), cudnn_workspace->conv_desc, cudnn_workspace->conv_bwd_data_algo,
        cudnn_workspace->d_workspace, cudnn_workspace->workspace_size, &(cudnn_workspace->zero), cudnn_workspace->input_desc, dX->data()));

      // weights backwards
      checkCudnnErrors(
        cudnnConvolutionBackwardFilter((*cudnn_workspace)(), &(cudnn_workspace->one), cudnn_workspace->input_desc, prev_layer->data(),
          cudnn_workspace->output_desc, next_layer_grad->data(),
          cudnn_workspace->conv_desc, cudnn_workspace->conv_bwd_filter_algo, cudnn_workspace->d_workspace,
          cudnn_workspace->workspace_size, &(cudnn_workspace->zero), cudnn_workspace->filter_desc, dW->data()));
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
      kernel = DeviceTensor<Scalar, 4, Device_, RowMajor>(v);
    }

    void set_bias(PtrTensorAdapter<Scalar, Device_>& v) override {
      bias = DeviceTensor<Scalar, 1, Device_, RowMajor>(v);
    }

  private:
    DeviceTensor<Scalar, 4, Device_, RowMajor> kernel, layer_output, dX, dW;
    DeviceTensor<Scalar, 1, Device_, RowMajor> bias, loss_by_bias_derivative;

    const int stride, dilation;
    const Padding2D padding;
    array<Index, 4> bias_broadcast;

    // we don't know the input dimension offhand, so default initialization
    std::shared_ptr<ConvolutionParams<4>> params;
    std::shared_ptr<CudnnWorkspace<Device_>> cudnn_workspace;
  };
} // namespace EigenSinn