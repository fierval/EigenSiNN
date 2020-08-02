#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include "ops/initializations.hpp"

namespace EigenSinn {

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  template <typename Scalar>
  class Conv2d : LayerBase {

  public:

    Conv2d(const array<Index, 4>& kernelDims, const Padding2D& _padding = { 0, 0 }, const Index _stride = 1) :
      kernel(kernelDims)
      , padding(_padding)
      , stride(_stride)
      {}


    // TODO: this needs to be implemented for real
    // Also strides and padding should be added
    void init() override {
      kernel = generate_xavier<Scalar, 4>(kernel.dimensions());
    }

    void init(const Tensor<Scalar, 4> _weights) {
      kernel = _weights;
    }

    void forward(std::any prev_layer) override {

      layer_output = convolve(std::any_cast<Tensor<Scalar, 4>&>(prev_layer), kernel, padding, stride);
    }

    // during the backward pass we get next_layer_grad alreayd flattened
    // TODO: prev_layer_output to come back flattened as well?
    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

      Tensor<Scalar, 4> prev_layer = std::any_cast<Tensor<Scalar, 4>&>(prev_layer_any);
      Tensor<Scalar, 2> next_layer_grad = std::any_cast<Tensor<Scalar, 2>&>(next_layer_grad_any);

      // flatten weights and kernel
      Tensor<Scalar, 2> unf_kernel = unfold_kernel(kernel);
      Tensor<Scalar, 2> x_col = im2col(prev_layer, kernel.dimensions(), padding, stride);

      // dX: kernel.T * dout
      ProductDims prod_dims = { IndexPair<int>(0, 0)};
      Tensor<Scalar, 2> dX_col = unf_kernel.contract(next_layer_grad, prod_dims);

      prod_dims = { IndexPair<int>(1, 1) };
      Tensor<Scalar, 2> dW_col = next_layer_grad.contract(x_col, prod_dims);

      dX = col2im(dX_col, kernel.dimensions(), prev_layer.dimensions(), padding, stride);
      dW = fold_kernel(dW_col, kernel.dimensions());
    }

    const std::any get_loss_by_input_derivative() override {
      return dX;
    }

    // feed to optimizer
    const std::any get_loss_by_weights_derivative() override {
      return dW;
    }

    const std::any get_output() {
      return layer_output;
    }

    Tensor<Scalar, 4>& get_weights() {
      return kernel;
    }

  private:
    Tensor<Scalar, 4> kernel, layer_output, dX, dW;
    const Index stride;
    const Padding2D padding;
  };
}