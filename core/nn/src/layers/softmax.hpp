#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, typename Device_ = DefaultDevice>
  class Softmax : public LayerBase<Device_> {
  public:
    Softmax(const Device_& _device = DefaultDevice()) :
      LayerBase(_device)
      , inited(false)
    {}

    void forward(std::any prev_layer_any) override {
      Tensor<Scalar, 2> prev_layer = from_any<Scalar, 2>(prev_layer_any);
      
      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

      Tensor<Scalar, 0> layer_max;
      layer_max.device(device) = prev_layer.maximum();

      Tensor<Scalar, 2> layer_max_broadcast(prev_layer.dimensions());
      layer_max_broadcast.setConstant(layer_max(0));

      exp_all.device(device) = (prev_layer - layer_max_broadcast).exp();

      Tensor<Scalar, 0> exp_sum = exp_all.sum();
      layer_output.device(device) = exp_all / exp_sum(0);
    }

    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {
      auto dout = from_any<Scalar, 2>(next_layer_grad_any);

      layer_grad.device(device) = next_layer_grad * layer_output * (cronecker_delta - layer_output);
    }

    std::any get_output() override {
      return layer_output;
    };

    std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };

  private:
    void init_cached(Eigen::Tensor<Scalar, 2>& prev_layer)
    {
      layer_output.resize(prev_layer.dimensions());
      layer_grad.resize(prev_layer.dimensions());
      exp_all.resize(prev_layer.dimensions());
    }

    bool inited;
    Tensor<Scalar, 2> layer_output, layer_grad, exp_all;
  };


}