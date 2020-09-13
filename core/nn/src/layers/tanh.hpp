#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Tanh : public LayerBase<Device_> {
  public:
    // leaky relu if necessary
    Tanh(const Device_& _device = DefaultDevice()) :
      LayerBase(_device)
      , inited(false)
    {}

    void forward(std::any prev_layer_any) override {
      Tensor<Scalar, Rank> prev_layer = from_any<Scalar, Rank>(prev_layer_any);

      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

      layer_output.device(device) = prev_layer.tanh();
    }


    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {
      auto next_layer_grad = from_any<Scalar, Rank>(next_layer_grad_any);

      layer_grad.device(device) = next_layer_grad * (ones - layer_output.pow(2.));
    }

    std::any get_output() override {
      return layer_output;
    };

    std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };


  private:
    void init_cached(Eigen::Tensor<Scalar, Rank>& prev_layer)
    {
      ones.resize(prev_layer.dimensions());
      ones.setConstant(1);

      layer_output.resize(prev_layer.dimensions());
      layer_grad.resize(prev_layer.dimensions());
    }

    bool inited;
    Tensor<Scalar, Rank> layer_output, layer_grad;
    Tensor<Scalar, Rank> ones;
  };


}