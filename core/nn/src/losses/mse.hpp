#pragma once

#include "layers/layer_base.hpp"

namespace EigenSinn {

  template<typename Scalar, Index Rank>
  class MseLoss : LayerBase {

    void forward(std::any prev_layer_any) override {};

    void backward(std::any prev_layer, std::any next_layer_grad) override {};

    const std::any get_output() override {};

  };
}