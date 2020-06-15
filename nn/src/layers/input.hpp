#pragma once

#include "layer_base.hpp"
#include "ops/linearops.hpp"

namespace EigenSinn {
  class InputLayer {

  public:
    InputLayer(LinearTensor& _layer_content, bool use_bias = false)
    {
      layer_content = _layer_content;
      if (use_bias) {
        layer_content = adjust_linear_bias(layer_content);
      }
    }

    LinearTensor& get_layer() {
      return layer_content;
    }

    const Index batch_size() {
      return layer_content.dimension(0);
    }

    const Index input_vector_dim() {
      return layer_content.dimension(1);
    }

  private:
    LinearTensor layer_content;
  };
}