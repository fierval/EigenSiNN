#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class LeakyReLU : public LayerBase<Scalar> {
  public:
    // leaky relu if necessary
    LeakyReLU(float _thresh = 0.01) :
      thresh(_thresh) {

    }

    void forward(LayerBase<Scalar>& prev_layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer.get_output());
      auto res = leaky_relu(x, thresh);

      layer_output = res.second;
      mask = res.first;
    }

    void backward(LayerBase<Scalar>& prev_layer, std::any next_layer_grad) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(next_layer_grad);

      layer_grad = leaky_relu_back(x, mask);
    }

    std::any get_output() override {
      return layer_output;
    };

    std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };


  private:
    float thresh;
    DeviceTensor<Scalar, Rank, Device_, Layout> mask;
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_grad;
  };

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class ReLU : public LeakyReLU<Scalar, Rank, Device_, Layout> {
  public:
    ReLU() : LeakyReLU(0) {}
  };

}