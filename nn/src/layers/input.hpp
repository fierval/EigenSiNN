#pragma once

#include "layer_base.hpp"

namespace EigenSinn {
  class InputLayer : public LayerBase {

  public:
    InputLayer(MatrixXd& _layer_conent, bool use_bias = false)
      : layer_content(_layer_conent) {
      if (use_bias) {
        adjust_linear_bias(layer_content);
      }
    }

    MatrixXd& get_layer() {
      return layer_content;
    }

    const Index batch_size() {
      return layer_content.rows();
    }

    const Index input_vector_dim() {
      return layer_content.cols();
    }

    void forward(MatrixXd& prev_layer) override {};

    void backward(const MatrixXd& prev_layer, const MatrixXd& next_layer_grad) override {};

  private:
    MatrixXd layer_content;
  };
}