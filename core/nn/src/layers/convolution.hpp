#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"

namespace EigenSinn {

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  template <typename Scalar, Index Rank = 4>
  class Conv2d : LayerBase {

  public:

    Conv2d(array<Index, Rank> kernelDims) : kernel(kernelDims) {}

    // TODO: this needs to be implemented for real
    // Also strides and padding should be added
    void init() override {
      kernel.setRandom<Eigen::internal::NormalRandomGenerator<float>>();
    }

    void init(const Tensor<Scalar, Rank> _weights) {
      kernel = _weights;
    }

    void forward(std::any prev_layer) override {

      layer_output = convolve_valid(std::any_cast<Tensor<Scalar, Rank>&>(prev_layer), kernel);
    }

    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

      Tensor<Scalar, Rank> prev_layer = std::any_cast<Tensor<Scalar, Rank>&>(prev_layer_any);
      Tensor<Scalar, Rank> next_layer_grad = std::any_cast<Tensor<Scalar, Rank>&>(next_layer_grad_any);

      // dL/dF = dL/dY * dY/dF
      // dL/dF = X conv dL/dY (used for gradient computation)
      // has kernel dim
      derivative_by_filter = convolve_valid(prev_layer, next_layer_grad);

      // dL/dX = dL/dY * dY/dX
      // dL/dX = F full_conv dL/dY (used for gradients chaining as next_layer_grad)
      // has layer output dim
      array<bool, 4> rev_idx({ false, false, true, true });
      Tensor<Scalar, Rank> rev_kernel = kernel.reverse(rev_idx);

      derivative_by_input = convolve_full(rev_kernel, next_layer_grad);
    }

    const std::any get_loss_by_input_derivative() override {
      return derivative_by_input;
    }

    // feed to optimizer
    const std::any get_loss_by_weights_derivative() override {
      return derivative_by_filter;
    }

    const std::any get_output() {
      return layer_output;
    }

    Tensor<Scalar, Rank>& get_weights() {
      return kernel;
    }

  private:
    Tensor<Scalar, Rank> kernel, derivative_by_input, derivative_by_filter, layer_output;
  };
}