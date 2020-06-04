#pragma once

#include "layer_base.hpp"

class InputLayer : public LayerBase {

public:
  InputLayer(MatrixXd& _layer_conent, bool use_bias = false)
    : layer_content(_layer_conent) {
    if (use_bias) {
      adjust_linear_bias(layer_content);
    }
  }

  const MatrixXd& GetLayer() {
    return layer_content;
  }

private:
  MatrixXd layer_content;
};