#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include "ops/initializations.hpp"

namespace EigenSinn {

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  template <typename Scalar, int Layout = ColMajor, typename Device_ = DefaultDevice>
  class TransConv2d : public LayerBase<Scalar> {

  public:

    TransConv2d(const array<Index, 4>& kernelDims, const Padding2D& _padding = { 0, 0 }, const Index _stride = 1, const Index _dilation = 1)
      : kernel(kernelDims)
      , padding(_padding)
      , stride(_stride)
      , dilation(_dilation)
      , bias(kernelDims[1])
      , bias_broadcast({ 0, 0, 0, 0 })
      , loss_by_bias_derivative(kernelDims[0])
    {}


    // TODO: this needs to be implemented for real
    // Also strides and padding should be added
    void init() override {

      // Wrapping the pointer and moving it to the tensor keeping in mind the GPU device
      kernel = generate_xavier<Scalar, 4, Layout, Device_>(kernel.dimensions());
      bias.setZero();
    }

    void init(const Tensor<Scalar, 4, Layout>& _weights) {
      init();

      kernel = _weights;
    }

    void forward(LayerBase<Scalar>& prev_layer_any) override {

      DeviceTensor<Device_, Scalar, 4, Layout> input(prev_layer_any.get_output());
      DeviceTensor<Device_, Scalar, 2, Layout> inp_reshaped = unfold_conv_res<Scalar, Layout, Device_>(input);

      
      DeviceTensor<Device_, Scalar, 4, Layout> dilated = dilate_tensor(kernel, dilation);
      DeviceTensor<Device_, Scalar, 2, Layout> unf_dilated = unfold_kernel(dilated);

      ProductDims prod_dims = { IndexPair<int>(0, 0) };
      DeviceTensor<Device_, Scalar, 2, Layout>  X_col(unf_dilated.dimension(1), inp_reshaped.dimension(1));
      X_col.view() = unf_dilated->contract(*inp_reshaped, prod_dims);
      
      DSizes<Index, 4> out_dims = get_output_dimensions(*input, kernel.dimensions(), padding, stride, dilation, true);

      layer_output = col2im(X_col, dilated.dimensions(), out_dims, padding, stride);
      //add bias to each channel
      auto dims = layer_output.dimensions();
      bias_broadcast = { out_dims[0], 1, out_dims[2], out_dims[3] };

      // one bias per filter
      layer_output.view() += bias->reshape(array<Index, 4>{ 1, kernel.dimension(1), 1, 1 }).broadcast(bias_broadcast);

    }

    void backward(LayerBase<Scalar>& prev_layer_any, std::any next_layer_grad_any) override {

      DeviceTensor<Device_, Scalar, 4, Layout> prev_layer(prev_layer_any.get_output());
      DeviceTensor<Device_, Scalar, 4, Layout> next_layer_grad(next_layer_grad_any);

      DeviceTensor<Device_, Scalar, 2, Layout> dout = unfold_conv_res<Scalar, Layout, Device_>(next_layer_grad);

      // flatten weights and kernel
      DeviceTensor<Device_, Scalar, 2, Layout> unf_kernel = unfold_kernel(kernel);
      DeviceTensor<Device_, Scalar, 2, Layout> x_col = im2col<Scalar, 4, Layout, Device_>(prev_layer, kernel.dimensions(), padding, stride, dilation);

      // dX: kernel.T * dout
      DeviceTensor<Device_, Scalar, 4, Layout> dilated = dilate_tensor(kernel, dilation);
      DeviceTensor<Device_, Scalar, 2, Layout> unf_dilated = unfold_kernel(dilated);
      ProductDims prod_dims = { IndexPair<int>(0, 0) };
      DeviceTensor<Device_, Scalar, 2, Layout>  dX_col(unf_dilated.dimension(1), dout.dimension(1));
      dX_col.view() = unf_dilated->contract(*dout, prod_dims);

      // dW: dout * x_col.T
      prod_dims = { IndexPair<int>(1, 1) };
      DeviceTensor<Device_, Scalar, 2, Layout>  dW_col(dout.dimension(0), x_col.dimension(0));
      dW_col.view() = dout->contract(*x_col, prod_dims);

      dX = col2im(dX_col, dilated.dimensions(), prev_layer.dimensions(), padding, stride);
      dW = fold_kernel(dW_col, kernel.dimensions());

      //bias
      if (!loss_by_bias_derivative) {
        loss_by_bias_derivative.resize(next_layer_grad.dimension(1));
      }

      loss_by_bias_derivative.view() = next_layer_grad->sum(array<Index, 3>{0, 2, 3});
    }

    std::any get_loss_by_input_derivative() override {
      return dX;
    }

    // feed to optimizer
    std::any get_loss_by_weights_derivative() override {
      return dW;
    }

    std::any get_output() override {
      return layer_output;
    }

    std::any get_weights() override {
      return kernel;
    }

    std::any get_bias() override {
      return bias;
    }

    std::any get_loss_by_bias_derivative() override {
      return loss_by_bias_derivative;
    }

    void set_weights(const DeviceTensor<Device_, Scalar, 4, Layout>& v) {
      kernel = v;
    }

    void set_bias(const DeviceTensor<Device_, Scalar, 1, Layout>& v) {
      bias = v;
    }

  private:
    DeviceTensor<Device_, Scalar, 4, Layout> kernel, layer_output, dX, dW;
    DeviceTensor<Device_, Scalar, 1, Layout> bias, loss_by_bias_derivative;

    const Index stride, dilation;
    const Padding2D padding;
    array<Index, 4> bias_broadcast;
  };
}