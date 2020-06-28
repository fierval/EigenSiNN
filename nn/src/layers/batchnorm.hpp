#pragma once

#include "layer_base.hpp"
#include <ops/batchnorm.hpp>
#include <ops/convolutions.hpp>

using namespace  Eigen;

namespace EigenSinn {

  template <int Rank = 2>
  class BatchNormalizationLayer : LayerBase {
  public:

    BatchNormalizationLayer(float _eps = 0.001, float _momentum = 0.99, int channels = 1)
      : beta(1, channels)
      , gamma(1, channels)
      , momentum(_momentum)
      , eps(_eps)
      , running_variance()
      , running_mean() {

    }

    void init() override {
      beta.setZero();
      gamma.setZero();

    }

    void forward(std::any prev_layer) override {


    }

    void backward(std::any prev_layer, std::any next_layer_grad) override {

    }

    Tensor<float, Rank>& get_output() {
      return layer_output;
    }

  private:
    Tensor<float, Rank> layer_output;

    TensorSingleDim gamma, beta, running_mean, running_variance;
    float momentum, eps;

  };

}