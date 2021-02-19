#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  class LeakyReLU : public LayerBase<Scalar> {
  public:
    // leaky relu if necessary
    LeakyReLU(float _thresh = 0.01) :
      thresh(_thresh) {

    }

    void forward(LayerBase<Scalar>& prev_layer) override {

      DeviceTensor<Device_, Scalar, Rank, Layout> x(prev_layer.get_output());
      auto res = leaky_relu<Scalar, Rank, Layout, Device_>(x, thresh);

      layer_output = res.second;
      mask = res.first;
    }

    void backward(LayerBase<Scalar>& prev_layer, std::any next_layer_grad) override {

      DeviceTensor<Device_, Scalar, Rank, Layout> x(next_layer_grad);

      layer_grad = leaky_relu_back<Scalar, Rank, Layout, Device_>(x, mask);
    }

    std::any get_output() override {
      return layer_output;
    };

    std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };


  private:
    float thresh;
    DeviceTensor<Device_, Scalar, Rank, Layout> mask;
    DeviceTensor<Device_, Scalar, Rank, Layout> layer_output, layer_grad;
  };

  template<typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  class ReLU : public LeakyReLU<Scalar, Rank, Layout, Device_> {
  public:
    ReLU() : LeakyReLU(0) {}
  };

}