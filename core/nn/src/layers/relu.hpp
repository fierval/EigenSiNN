#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank>
  class ReLU : LayerBase {
  public:
    // leaky relu if necessary
    ReLU() {

    }
    void init() override {};

    void forward(std::any prev_layer_any) override {
      auto res = relu<Scalar, Rank>(from_any<Scalar, Rank>(prev_layer_any), thresh);

      layer_output = res.second;
      mask = res.first;
    }

    void backward(std::any prev_layer, std::any next_layer_grad) override {

    }

    const std::any get_output() override {
      return layer_output;
    };

    const std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };


  private:
    Tensor<byte, 1> mask;
    Tensor<Scalar, Rank> layer_output, layer_grad;
  };
}