#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Sigmoid : public LayerBase<Scalar, Device_> {
  public:
    // leaky relu if necessary
    Sigmoid(Dispatcher<Device_>& _device =  LayerBase::default_dispatcher) :
      LayerBase(_device)
      , inited(false)
    {}

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) override {

      set_dims(prev_layer_any);
      TensorMap<Tensor<Scalar, Rank>> prev_layer(prev_layer_any.get_output(), vector2array< Rank>(in_dims));
      
      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

      layer_output.device(dispatcher.get_device()) = prev_layer.sigmoid();
    }


    void backward(LayerBase<Scalar, Device_>& prev_layer_any, Scalar * next_layer_grad_any) override {
      TensorMap<Tensor<Scalar, Rank>> next_layer_grad(next_layer_grad_any, vector2array< Rank>(out_dims));

      layer_grad.device(dispatcher.get_device()) = next_layer_grad * layer_output * (ones - layer_output);
    }

    Scalar * get_output() override {
      return layer_output.data();
    };

    Scalar * get_loss_by_input_derivative() override {
      return layer_grad.data();
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