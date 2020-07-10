#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"

namespace EigenSinn {

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  class ConvolutionalLayer : LayerBase {

  public:
    ConvolutionalLayer(Index batch_size, Index kernel_width, Index kernel_height, Index in_channels, Index out_channels) 
      : kernel(out_channels, kernel_width, kernel_height, in_channels),
      derivative_by_filter(out_channels, kernel_width, kernel_height, in_channels)
    {
      
    }

    // TODO: this needs to be implemented for real
    // Also strides and padding should be added
    void init() override {
      kernel.setRandom<Eigen::internal::NormalRandomGenerator<float>>();
    }

    void forward(std::any prev_layer) override {

      layer_output = convolve_valid(std::any_cast<ConvTensor&>(prev_layer), kernel);
    }

    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {
      
      ConvTensor prev_layer = std::any_cast<ConvTensor&>(prev_layer_any);
      ConvTensor next_layer_grad = std::any_cast<ConvTensor&>(next_layer_grad_any);

      // dL/dF = dL/dY * dY/dF
      // dL/dF = X conv dL/dY (used for gradient computation)
      // has kernel dim
      derivative_by_filter = convolve_valid(prev_layer, next_layer_grad);

      // dL/dX = dL/dY * dY/dX
      // dL/dX = F full_conv dL/dY (used for gradients chaining as next_layer_grad)
      // has layer output dim
      array<bool, 4> rev_idx({ false, true, false, false });
      ConvTensor rev_kernel = kernel.reverse(rev_idx);
      
      derivative_by_input = convolve_full(rev_kernel, next_layer_grad);
    }

    ConvTensor& get_loss_by_input_derivative() {
      return derivative_by_input;
    }

    // feed to optimizer
    ConvTensor& get_loss_by_weights_derivative() {
      return derivative_by_filter;
    }

    const std::any get_output() {
      return layer_output;
    }

    ConvTensor& get_weights() {
      return kernel;
    }

  private:
    ConvTensor kernel, derivative_by_input, derivative_by_filter, layer_output;
  };
}