#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank>
  class LeakyReLU : LayerBase {
  public:
    // leaky relu if necessary
    LeakyReLU(float _thresh) : thresh(_thresh) {

    }
    void init() override {};

    void forward(std::any prev_layer_any) override {
      auto res = leaky_relu(from_any<Scalar, Rank>(prev_layer_any), thresh);

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
    float thresh;
    Tensor<float, 1> mask;
    Tensor<Scalar, Rank> layer_output, layer_grad;
  };

  template<typename Scalar, Index Rank>
  class ReLU : public LeakyReLU<Scalar, Rank> {
  public:
    ReLU() : LeakyReLU(0) {}
  };

}