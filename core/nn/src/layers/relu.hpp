#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class LeakyReLU : public LayerBase<Scalar, Device_> {
  public:
    // leaky relu if necessary
    LeakyReLU(float _thresh = 0.01, Dispatcher<Device_>& _device =  LayerBase::default_dispatcher) :
      LayerBase(_device)
      ,thresh(_thresh) {

    }

    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      if (are_dims_unset(prev_layer.get_out_dims())) {
        set_dims(prev_layer.get_out_dims(), prev_layer.get_out_dims());
      }

      TensorMap<Tensor<Scalar, Rank>> x(prev_layer.get_output(), vector2array<int, Rank>(in_dims));
      auto res = leaky_relu(x, thresh);

      layer_output = res.second;
      mask = res.first;
    }

    void backward(LayerBase<Scalar, Device_>& prev_layer, Scalar * next_layer_grad) override {

      TensorMap<Tensor<Scalar, Rank>> x(next_layer_grad, vector2array<int, Rank>(out_dims));

      layer_grad = leaky_relu_back(x, mask, dispatcher.get_device());
    }

    Scalar * get_output() override {
      return layer_output.data();
    };

    Scalar * get_loss_by_input_derivative() override {
      return layer_grad.data();
    };


  private:
    float thresh;
    Tensor<Scalar, Rank> mask;
    Tensor<Scalar, Rank> layer_output, layer_grad;
  };

  template<typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class ReLU : public LeakyReLU<Scalar, Rank, Device_> {
  public:
    ReLU(Dispatcher<Device_>& _device = LayerBase::default_dispatcher) : LeakyReLU(0, _device) {}
  };

}