#pragma once

#include "layer_base.hpp"
#include <ops/batchnorm.hpp>

using namespace  Eigen;

namespace EigenSinn {

  // NCHW format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class BatchNormalizationLayer : public LayerBase<Device_> {
  public:

    BatchNormalizationLayer(Index num_features, float _eps = 1e-5, float _momentum = 0.9, bool _is_training = true, const Device_& _device = DefaultDevice())
      : LayerBase(_device)
      , beta(num_features)
      , gamma(num_features)
      , dbeta(num_features)
      , dgamma(num_features)
      , momentum(_momentum)
      , eps(_eps)
      , running_variance(num_features)
      , running_mean(num_features)
      , mu()
      , var()
      , is_training(_is_training) {

    }

    void init() override {
      beta.setZero();
      gamma.setConstant(1.);
      running_variance.setZero();
      running_mean.setZero();
    }

    void init(TensorSingleDim<Scalar>& _beta, TensorSingleDim<Scalar> _gamma) {
      this->init();
      beta = _beta;
      gamma = _gamma;
    }
    void forward(std::any prev_layer_any) override {

      Tensor<Scalar, Rank> prev_layer = from_any<Scalar, Rank>(prev_layer_any);

      std::tie(layer_output, xhat, running_mean, running_variance, mu, var) =
        batch_norm(prev_layer, gamma, beta, eps, momentum, running_mean, running_variance, is_training, device);
    }

    // see https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    // for derivations
    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

      Tensor<Scalar, Rank> prev_layer = from_any<Scalar, Rank>(prev_layer_any);
      Tensor<Scalar, Rank> dout = from_any<Scalar, Rank>(next_layer_grad_any);

      array<Index, Rank - 1> reduction_dims;
      array<Index, Rank> broadcast_dims;

      float total_channel = 1.;
      for (int i = 0; i < Rank; i++) {
        if (i == (int)ImageDims::channel) {
          continue;
        }
        total_channel *= prev_layer.dimension(i);
      }

      std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims(dout);

      //broadcast values
      Tensor<Scalar, Rank> broadcast_mean = broadcast_as_last_dim(mu, broadcast_dims, device);
      Tensor<Scalar, Rank> broadcast_var = broadcast_as_last_dim(var, broadcast_dims, device);
      Tensor<Scalar, Rank> xmu = (prev_layer - broadcast_mean);

      // Step 9
      // dbeta = sum(dout, reduced by all dims except channel)
      dbeta = dout.sum(reduction_dims);

      // Step 8
      // dgamma = sum (dout * y, reduced by all dims except channel)
      Tensor<Scalar, Rank> gamma_broad = broadcast_as_last_dim(gamma, broadcast_dims, device);
      Tensor<Scalar, Rank> dxhat(dout.dimensions());

      dxhat.device(device) = dout * gamma_broad;
      dgamma.device(device) = (dout * xhat).sum(reduction_dims);

      // Step 7
      // d_inv_std
      //TensorSingleDim d_inv_std = (dxhat * xmu).sum(reduction_dims);
      Tensor<Scalar, Rank> dxmu1(dxhat.dimensions());
      dxmu1.device(device) = dxhat * (1. / (broadcast_var + eps).sqrt());

      // Step 6
      // TensorSingleDim d_std = -d_inv_std / (running_var + eps);

      // Step 5
      Tensor<Scalar, 1> d_var(var.dimensions());
      d_var.device(device) = -0.5 * (dxhat * xmu).sum(reduction_dims) / (var + eps).pow(3. / 2.);

      // Step 4
      Tensor<Scalar, Rank> d_sq(dout.dimensions());
      d_sq.device(device) = 1. / total_channel * broadcast_as_last_dim<Scalar, Rank>(d_var, broadcast_dims, device);

      // Step 3
      Tensor<Scalar, Rank> dxmu2(d_sq.dimensions());
      dxmu2.device(device) = 2 * xmu * d_sq;

      // step 2
      Tensor<Scalar, Rank> dx1(dxmu1.dimensions());
      dx1.device(device) = dxmu1 + dxmu2;
      Tensor<Scalar, 1> dmu(dout.dimension(1));

      dmu.device(device) = -dx1.sum(reduction_dims);

      // step 1
      Tensor<Scalar, Rank> dx2(dout.dimensions());
      dx2.device(device) = 1. / total_channel * broadcast_as_last_dim<Scalar, Rank>(dmu, broadcast_dims, device);

      // step 0
      layer_gradient.resize(dout.dimensions());
      layer_gradient.device(device) = dx1 + dx2;
    }

    std::any get_output() {

      return layer_output;
    }

    std::any get_loss_by_input_derivative() {
      return layer_gradient;
    }

    std::any get_loss_by_weights_derivative() override {
      return dgamma;
    }

    std::any get_loss_by_bias_derivative() override {
      return dbeta;
    }

    inline void SetTraining(bool training) {
      is_training = training;
    }

    inline bool IsTraining() {
      return is_training;
    }

    std::any get_weights() override {
      return gamma;
    }

    std::any get_bias() override {
      return beta;
    }

    void set_weights(const std::any _weights) override {
      gamma = from_any<Scalar, 1>(_weights);
    }

    void set_bias(const std::any _bias) override {
      beta = from_any<Scalar, 1>(_bias);
    }



  private:
    Tensor<Scalar, Rank> layer_output, layer_gradient, xhat;

    TensorSingleDim<Scalar> gamma, beta, running_mean, running_variance, mu, var;
    TensorSingleDim<Scalar> dbeta, dgamma;
    float momentum, eps;
    bool is_training;

  };

}