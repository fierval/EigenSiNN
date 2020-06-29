#pragma once

#include "layer_base.hpp"
#include <ops/batchnorm.hpp>

using namespace  Eigen;

namespace EigenSinn {

  // NHWC format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <Index Rank = 2>
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
      std::tie(layer_output, xhat, running_mean, running_variance) = 
        batch_norm(prev_layer, gamma, beta, eps, momentum, running_mean, running_variance);
    }

    // see https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    // for derivations
    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

      NnTensor<Rank> prev_layer = std::any_cast<NnTensor<Rank>&>(prev_layer_any);
      NnTensor<Rank> dout = std::any_cast<NnTensor<Rank>&>(next_layer_grad_any);

      array<int, Rank - 1> reduction_dims;
      array<int, Rank> broadcast_dims;

      float total_channel;
      for (total_channel = 1., int i = 0; i < Rank - 1; i++) {
        total_channel *= prev_layer.dimension(i);
      }

      std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims(dout);

      //broadcast values
      NnTensor<Rank> broadcast_mean = broadcast_as_last_dim(running_mean, broadcast_dims);
      NnTensor<Rank> broadcast_var = broadcast_as_last_dim(running_variance, broadcast_dims);
      NnTensor<Rank> xmu = (prev_layer - broadcast_mean);

      // Step 9
      // dbeta = sum(dout, reduced by all dims except channel)
      dbeta = dout.sum(reduction_dims);

      // Step 8
      // dgamma = sum (dout * y, reduced by all dims except channel)
      NnTensor<Rank> gamma_broad = broadcast_as_last_dim(gamma, broadcast_dims);
      NnTensor<Rank> dxhat = dout * gamma_broad;
      NnTensor<Rank> dgamma = (dout * xhat).sum(reduction_dims);
      NnTensor<Rank> dx_hat = dout * gamma;

      // Step 7
      // d_inv_std
      //TensorSingleDim d_inv_std = (dxhat * xmu).sum(reduction_dims);
      NnTensor<Rank> dxmu1 = dxhat * (1. / (broadcast_var + eps).sqrt());

      // Step 6
      // TensorSingleDim d_std = -d_inv_std / (running_var + eps);

      // Step 5
      TensorSingleDim d_var = -0.5 * (dxhat * xmu).sum(reduction_dims) / (running_var + eps).pow(3./2.);

      // Step 4
      NnTensor<Rank> d_sq = (1 - momentum) * broadcast_as_last_dim(d_var / total_channel, broadcast_dims);

      // Step 3
      NnTensor<Rank> dxmu2 = 2 * xmu * d_sq;

      // step 2
      NnTensor<Rank> dx1 = dxmu1 + dxmu2;
      TensorSingleDim dmu = -dx1.sum(reduction_dims);

      // step 1
      NnTensor<Rank> dx2 = (1 - momentum) * broadcast_as_last_dim(dmu / total_channel, broadcast_dims);

      // step 0
      layer_gradient = dx1 + dx2;

    }

    template <Index Rank = 2>
    NnTensor<Rank>& get_output() {
      return layer_output;
    }

  private:
    NnTensor<Rank> layer_output, layer_gradient, xhat;

    TensorSingleDim gamma, beta, running_mean, running_variance;
    TensorSingleDim dbeta, dgamma;
    float momentum, eps;

  };

}