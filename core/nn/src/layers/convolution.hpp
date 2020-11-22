#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include "ops/initializations.hpp"

namespace EigenSinn {

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  template <typename Scalar, int Layout = ColMajor, typename Device_ = DefaultDevice>
  class Conv2d : public LayerBase<Scalar> {

  public:

    Conv2d(const array<Index, 4>& kernelDims, const Padding2D& _padding = { 0, 0 }, const Index _stride = 1,
      Dispatcher<Device_>& _device = LayerBase::default_dispatcher) :
        kernel(kernelDims)
      , padding(_padding)
      , stride(_stride)
      , bias(kernelDims[0])
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

    void init(const Tensor<Scalar, 4>& _weights) {
      init();

      kernel = _weights;
    }

    void forward(LayerBase<Scalar>& prev_layer_any) override {

      DeviceTensor<Device_, Scalar, 4, Layout> prev_layer(prev_layer_any.get_output(), vector2array<4>(prev_layer_any.get_out_dims()));

      layer_output = convolve<Scalar, 4, Device_, Layout, Device_>(prev_layer, kernel, padding, stride);

      //add bias to each channel
      auto dims = layer_output.dimensions();
      bias_broadcast = { dims[0], 1, dims[2], dims[3] };

      // one bias per filter
      layer_output.view() += bias->reshape(array<Index, 4>{ 1, kernel.dimension(0), 1, 1 }).broadcast(bias_broadcast);

    }

    void backward(LayerBase<Scalar>& prev_layer_any, std::any next_layer_grad_any) override {

      DeviceTensor<Device_, Scalar, 4, Layout> prev_layer(prev_layer_any.get_output());
      DeviceTensor<Device_, Scalar, 4, Layout> next_layer_grad(next_layer_grad_any);

      DeviceTensor<Device_, Scalar, 2, Layout> dout = unfold_conv_res<Scalar, Layout, Device_>(next_layer_grad);

      // flatten weights and kernel
      DeviceTensor<Device_, Scalar, 2, Layout> unf_kernel = unfold_kernel(kernel);
      DeviceTensor<Device_, Scalar, 2, Layout> x_col = im2col<Device_, Scalar, 4, Layout>(prev_layer, kernel.dimensions(), padding, stride);

      // dX: kernel.T * dout
      ProductDims prod_dims = { IndexPair<int>(0, 0) };
      DeviceTensor<Device_, Scalar, 2, Layout>  dX_col(unf_kernel.dimension(1), dout.dimension(1));
      dX_col = unf_kernel->contract(dout, prod_dims);

      // dW: dout * x_col.T
      prod_dims = { IndexPair<int>(1, 1) };
      DeviceTensor<Device_, Scalar, 2, Layout>  dW_col(dout.dimension(0), x_col.dimension(0));
      dW_col = dout->contract(x_col, prod_dims);

      dX = col2im(dX_col, kernel.dimensions(), prev_layer.dimensions(), padding, stride);
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

  private:
    DeviceTensor<Device_, Scalar, 4, Layout> kernel, layer_output, dX, dW;
    DeviceTensor<Device_, Scalar, 1, Layout> bias, loss_by_bias_derivative;

    const Index stride;
    const Padding2D padding;
    array<Index, 4> bias_broadcast;
  };
}