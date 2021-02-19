#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  class Tanh : public LayerBase<Scalar> {
  public:
    // leaky relu if necessary
    Tanh() 
      : inited(false)  {}

    void forward(LayerBase<Scalar>& prev_layer_any) override {

      DeviceTensor<Device_, Scalar, Rank, Layout> prev_layer(prev_layer_any.get_output());

      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

      layer_output.view() = prev_layer->tanh();
    }


    void backward(LayerBase<Scalar>& prev_layer_any, std::any next_layer_grad_any) override {

      DeviceTensor<Device_, Scalar, Rank, Layout> next_layer_grad(next_layer_grad_any);

      layer_grad.view() = *next_layer_grad * (*ones - layer_output->pow(2.));
    }

    std::any get_output() override {
      return layer_output;
    };

    std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };


  private:
    void init_cached(const DeviceTensor<Device_, Scalar, Rank, Layout>& prev_layer)
    {
      ones.resize(prev_layer.dimensions());
      ones.setConstant(1);

      layer_output.resize(prev_layer.dimensions());
      layer_grad.resize(prev_layer.dimensions());
    }

    bool inited;
    DeviceTensor<Device_, Scalar, Rank, Layout> ones, layer_output, layer_grad;
  };


}