#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"

namespace EigenSinn {
  typedef Eigen::Tensor<float, 1> ConvBias;

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  class ConvolutionalLayer : LayerBase {

  public:
    ConvolutionalLayer(index batch_size, Index kernel_width, Index kernel_height, Index in_channels, Index out_channels) 
      : kernel(out_channels, kernel_width, kernel_height, in_channels),
      derivative_by_filter(out_channels, kernel_width, kernel_height, in_channels)
    {
      
    }

    // TODO: this needs to be implemented for real
    // Also strides and padding should be added
    void init() {
      kernel.setRandom<Eigen::internal::NormalRandomGenerator<float>>();
      layer_output.resize(batch_size, )
    }

    void forward(std::any prev_layer) override {

      layer_output = convolve_valid(std::any_cast<ConvTensor&>(prev_layer), kernel);
    }

    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {
      
      ConvTensor prev_layer = std::any_cast<ConvTensor&>(prev_layer_any);
      ConvTensor next_layer_grad = std::any_cast<ConvTensor&>(next_layer_grad_any);

      derivative_by_filter = convolve_valid(prev_layer, next_layer_grad);
      derivative_by_input = convolve_full(kernel.reverse({ false, true, false, false }).eval(), next_layer_grad);
    }

  private:
    ConvTensor kernel, derivative_by_input, derivative_by_filter, layer_output;
  };
}