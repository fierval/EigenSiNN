#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class LeakyReLU : public LayerBase<Device_> {
  public:
    // leaky relu if necessary
    LeakyReLU(float _thresh = 0.01, const Device_ & _device = DefaultDevice()) : 
      LayerBase(_device)
      ,thresh(_thresh) {

    }
    void init() override {};

    void forward(std::any prev_layer_any) override {
      auto res = leaky_relu(from_any<Scalar, Rank>(prev_layer_any), thresh);

      layer_output = res.second;
      mask = res.first;
    }

    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {
      layer_grad = leaky_relu_back(from_any<Scalar, Rank>(next_layer_grad_any), mask, device);
    }

    std::any get_output() override {
      return layer_output;
    };

    std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };


  private:
    float thresh;
    Tensor<Scalar, Rank> mask;
    Tensor<Scalar, Rank> layer_output, layer_grad;
  };

  template<typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class ReLU : public LeakyReLU<Scalar, Rank> {
  public:
    ReLU(const Device_& _device = DefaultDevice()) : LeakyReLU(0, _device) {}
  };

}