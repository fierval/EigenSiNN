#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include "ops/initializations.hpp"
#include "helpers/conv_params_bag.hpp"

#ifdef EIGEN_USE_GPU
#include "cudnn/cudnn_workspace.hpp"
#endif

namespace EigenSinn {

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class TransConv2d : public LayerBase<Scalar, Device_> {

  public:

    TransConv2d(const array<Index, 4>& kernelDims, const Padding2D& _padding = { 0, 0 }, const Index _stride = 1, const Index _dilation = 1)
      : kernel(kernelDims)
      , padding(_padding)
      , stride(_stride)
      , dilation(_dilation)
      , bias(kernelDims[1])
      , bias_broadcast({ 0, 0, 0, 0 })
      , loss_by_bias_derivative(kernelDims[1])
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

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) override {

      DeviceTensor<Scalar, 4, Device_, Layout> input(prev_layer_any.get_output());

      if (!params || params->check(input.dimensions())) {
        params = std::make_shared<ConvolutionParams<4>>(input.dimensions(), kernel.dimensions(), padding, stride, dilation, true, is_cudnn);
        layer_output.resize(params->orig_dims());
        dX.resize(params->output_dims());
        dW.resize(kernel.dimensions());
      }

      DSizes<Index, 4> out_dims = params->orig_dims();
      //add bias to each channel
      bias_broadcast = { out_dims[0], 1, out_dims[2], out_dims[3] };

#ifdef __INTELLISENSE__
#define EIGEN_USE_GPU
#endif

#ifdef EIGEN_USE_GPU
      if (is_cudnn && !cudnn_workspace) {
        cudnn_workspace = std::make_shared<CudnnWorkspace>(*params);
      }

      if (is_cudnn) {
        // data forward
        checkCudnnErrors(cudnnConvolutionBackwardData(CudnnWorkspace::cudnn(), &(cudnn_workspace->one), cudnn_workspace->filter_desc, kernel->data(),
          cudnn_workspace->output_desc, input->data(), cudnn_workspace->conv_desc, cudnn_workspace->conv_bwd_data_algo,
          cudnn_workspace->d_workspace, cudnn_workspace->workspace_size, &(cudnn_workspace->zero), cudnn_workspace->input_desc, layer_output->data()));

      }
      else {
#endif // EIGEN_USE_GPU

        DeviceTensor<Scalar, 2, Device_, Layout> inp_reshaped = unfold_conv_res<Scalar, Device_, Layout>(input);

        DeviceTensor<Scalar, 4, Device_, Layout> dilated = dilate_tensor(kernel, dilation);
        DeviceTensor<Scalar, 2, Device_, Layout> unf_dilated = unfold_kernel(dilated);

        // transposed convolution: kernel.T * x_col
        ProductDims prod_dims = { IndexPair<int>(0, 0) };
        DeviceTensor<Scalar, 2, Device_, Layout>  X_col(unf_dilated.dimension(1), inp_reshaped.dimension(1));
        X_col.view() = unf_dilated->contract(*inp_reshaped, prod_dims);

        // re-format into the image of output dimensions
        col2im(X_col, layer_output, *params, true);
#ifdef EIGEN_USE_GPU
      }
#endif

      // one bias per filter
      layer_output.view() += bias->reshape(array<Index, 4>{ 1, kernel.dimension(1), 1, 1 }).broadcast(bias_broadcast);

    }

    void backward(LayerBase<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, 4, Device_, Layout> prev_layer(prev_layer_any.get_output());
      DeviceTensor<Scalar, 4, Device_, Layout> next_layer_grad(next_layer_grad_any);

      DeviceTensor<Scalar, 2, Device_, Layout> dout = im2col<Scalar, 4, Device_, Layout>(next_layer_grad, *params);

      // dX: kernel.T.T * dout which is just a regular convolution
      dX = convolve<Scalar, 4, Layout, Device_>(next_layer_grad, kernel, *params);

      // dW: (dout * x_col.T.T)
      DeviceTensor<Scalar, 2, Device_, Layout> inp_reshaped = unfold_conv_res<Scalar, Device_, Layout>(prev_layer);
      ProductDims prod_dims = { IndexPair<int>(1, 1) };
      DeviceTensor<Scalar, 2, Device_, Layout>  dW_col(inp_reshaped.dimension(0), dout.dimension(0));
      dW_col.view() = inp_reshaped->contract(*dout, prod_dims);

      dW = fold_kernel(dW_col, kernel.dimensions());

      //bias
      loss_by_bias_derivative.view() = next_layer_grad->sum(array<Index, 3>{0, 2, 3});
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
    std::shared_ptr<ConvolutionParams<4>> params;

#ifdef EIGEN_USE_GPU
    std::shared_ptr<CudnnWorkspace> cudnn_workspace;
#endif
    bool is_cudnn;
  };
}