#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include "ops/initializations.hpp"

namespace EigenSinn {

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  template <typename Scalar, typename Device_ = DefaultDevice>
  class Conv2d : public LayerBase<Device_> {

  public:

    Conv2d(const array<Index, 4>& kernelDims, const Padding2D& _padding = { 0, 0 }, const Index _stride = 1,
      Dispatcher<Device_>& _device = LayerBase::default_dispatcher) :
      LayerBase(_device)
      , kernel(kernelDims)
      , padding(_padding)
      , stride(_stride)
      , bias(kernelDims[0])
      , bias_broadcast({ 0, 0, 0, 0 })
      , loss_by_bias_derivative(kernelDims[0])
    {}


    // TODO: this needs to be implemented for real
    // Also strides and padding should be added
    void init() override {
      kernel = generate_xavier<Scalar, 4>(kernel.dimensions(), dispatcher.get_device());
      bias.setZero();
    }

    void init(const Tensor<Scalar, 4> _weights) {
      kernel = _weights;
      bias.setZero();
    }

    void forward(std::any prev_layer_any) override {

      if (are_dims_unset())
      {
        set_in_dims(prev_layer.ge)
      }

      Tensor<Scalar, 4> prev_layer = std::any_cast<Tensor<Scalar, 4>&>(prev_layer_any);
      layer_output = convolve(prev_layer, kernel, padding, stride, dispatcher.get_device());


      //add bias to each channel
      auto dims = layer_output.dimensions();
      bias_broadcast = { dims[0], 1, dims[2], dims[3] };

      // one bias per filter
      Tensor<Scalar, 4> reshaped = bias.reshape(array<Index, 4>{ 1, kernel.dimension(0), 1, 1 });
      layer_output.device(dispatcher.get_device()) += reshaped.broadcast(bias_broadcast);

    }

    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

      Tensor<Scalar, 4> prev_layer = std::any_cast<Tensor<Scalar, 4>&>(prev_layer_any);
      Tensor<Scalar, 4> next_layer_grad = std::any_cast<Tensor<Scalar, 4>&>(next_layer_grad_any);

      Tensor<Scalar, 2> dout = unfold_conv_res(next_layer_grad);

      // flatten weights and kernel
      Tensor<Scalar, 2> unf_kernel = unfold_kernel(kernel);
      Tensor<Scalar, 2> x_col = im2col(prev_layer, kernel.dimensions(), padding, stride, dispatcher.get_device());

      // dX: kernel.T * dout
      ProductDims prod_dims = { IndexPair<int>(0, 0) };
      Tensor<Scalar, 2> dX_col(unf_kernel.dimension(1), dout.dimension(1));
      dX_col = unf_kernel.contract(dout, prod_dims);

      prod_dims = { IndexPair<int>(1, 1) };
      Tensor<Scalar, 2> dW_col(dout.dimension(0), x_col.dimension(0));
      dW_col = dout.contract(x_col, prod_dims);

      dX = col2im(dX_col, kernel.dimensions(), prev_layer.dimensions(), padding, stride);
      dW = fold_kernel(dW_col, kernel.dimensions());

      //bias
      loss_by_bias_derivative.resize(next_layer_grad.dimension(1));
      loss_by_bias_derivative.device(dispatcher.get_device()) = next_layer_grad.sum(array<Index, 3>{0, 2, 3});
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

    void set_weights(const std::any _weights) override {
      kernel = from_any<Scalar, 4>(_weights);
    }

    void set_bias(const std::any _bias) override {
      bias = from_any<Scalar, 1>(_bias);
    }


  private:
    Tensor<Scalar, 4> kernel, layer_output, dX, dW;
    Tensor<Scalar, 1> bias, loss_by_bias_derivative;

    const Index stride;
    const Padding2D padding;
    array<Index, 4> bias_broadcast;
  };
}