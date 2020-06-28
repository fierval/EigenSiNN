#pragma once

#include "layer_base.hpp"
#include <ops/batchnorm.hpp>
#include <ops/convolutions.hpp>

using namespace  Eigen;

namespace EigenSinn {

  // NHWC format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <int Rank = 2>
  class BatchNormalizationLayer : LayerBase {
  public:

    BatchNormalizationLayer(float _eps = 0.001, float _momentum = 0.99, int channels = 1)
      : beta(1, channels)
      , gamma(1, channels)
      , dbeta(1, channels)
      , dgamma(1, channels)
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
      std::tie(layer_output, running_mean, running_variance) = 
        batch_norm(prev_layer, gamma, beta, eps, momentum, running_mean, running_variance);
    }

    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

      Tensor<float, Rank> prev_layer = std::any_cast<Tensor<float, Rank>&>(prev_layer_any);
      Tensor<float, Rank> dout = std::any_cast<Tensor<float, Rank>&>(next_layer_grad_any);

      Eigen::array<int, Dim - 1> reduction_dims;
      Eigen::array<int, Dim> broadcast_dims;

      std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims(dout);

      // dbeta = sum(dout, reduced by all dims except channel)
      dbeta = dout.sum(reduction_dims);

      // dgamma = sum (dout * y, reduced by all dims except channel)
      Eigen::Tensor<float, Dim> gamma_broad = broadcast_as_last_dim(gamma, broadcast_dims);
      dgamma = (dout * layer_output).sum(reduction_dims);
    }

    Tensor<float, Rank>& get_output() {
      return layer_output;
    }

  private:
    Tensor<float, Rank> layer_output, layer_gradient;

    TensorSingleDim gamma, beta, running_mean, running_variance;
    TensorSingleDim dbeta, dgamma;
    float momentum, eps;

  };

}