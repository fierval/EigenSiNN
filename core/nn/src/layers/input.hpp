#pragma once

#include "layer_base.hpp"
#include "ops/linearops.hpp"

namespace EigenSinn {

  template <typename Scalar, Index Dim>
  class InputLayer : LayerBase {

  public:
    InputLayer(Tensor<Scalar, Dim>& _layer_content, bool use_bias = false)
    {
      layer_content = _layer_content;
      if (use_bias) {
        layer_content = adjust_linear_bias(layer_content);
      }
    }

    const std::any get_output() override {
      return layer_content;
    }

    const Index batch_size() {
      return layer_content.dimension(0);
    }

    const Index input_vector_dim() {
      return layer_content.dimension(1);
    }

    void forward(std::any prev_layer) override {
    }

    void backward(std::any prev_layer, std::any next_layer_grad) override {
    }

    void init() override {

    }

  private:
    LinearTensor layer_content;
  };
}