#pragma once

#include "layer_base.hpp"
#include <ops/batchnorm.hpp>
#include <device/device_tensor.hpp>

using namespace  Eigen;
using std::unique_ptr;
using std::make_unique;

namespace EigenSinn {

  // NCHW format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class BatchNormalizationLayer : public LayerBase<Scalar> {
  public:

    BatchNormalizationLayer(Index num_features, float _eps = 1e-5, float _momentum = 0.9, bool _is_training = true)
      : LayerBase<Scalar>()
      , momentum(_momentum)
      , eps(_eps)
      , is_training(_is_training)
      , beta(num_features)
      , dgamma(num_features)
      , dbeta(num_features)
      , gamma(num_features)
      , running_variance(num_features)
      , running_mean(num_features)
      , mu(num_features)
      , var(num_features) {
    }

    void init() override {
      beta.setZero();
      gamma.setConstant(1.);
      running_variance.setZero();
      running_mean.setZero();
    }

    void init(TensorSingleDim<Scalar>& _beta, TensorSingleDim<Scalar>& _gamma) {
      init();

      beta.set_from_host(_beta);
      gamma.set_from_host(_gamma);
    }

    void forward(LayerBase<Scalar>& prev_layer_base) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> prev_layer(prev_layer_base.get_output());

      if (!xhat) {
        DSizes<Index, Rank> dims = prev_layer.dimensions();
        layer_gradient.resize(dims);
        layer_output.resize(dims);
        xhat.resize(dims);
      }


      std::tie(layer_output, xhat, running_mean, running_variance, mu, var) =
        batch_norm(prev_layer, gamma, beta, eps, momentum, running_mean, running_variance, is_training);
    }

    // see https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    // for derivations
    void backward(LayerBase<Scalar>& prev_layer_any, std::any next_layer_grad_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> prev_layer(prev_layer_any.get_output());
      DeviceTensor<Scalar, Rank, Device_, Layout> dout(next_layer_grad_any);

      DSizes<Index, Rank - 1> reduction_dims;
      DSizes<Index, Rank> broadcast_dims;

      float total_channel = 1.;
      for (int i = 0; i < Rank; i++) {
        if (i == (int)ImageDims::channel) {
          continue;
        }
        total_channel *= prev_layer->dimension(i);
      }

      std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims<Scalar, Rank>(*dout);

      //broadcast values
      DeviceTensor<Scalar, Rank, Device_, Layout> broadcast_mean = broadcast_as_last_dim(mu, broadcast_dims);
      DeviceTensor<Scalar, Rank, Device_, Layout> broadcast_var = broadcast_as_last_dim(var, broadcast_dims);
      DeviceTensor<Scalar, Rank, Device_, Layout> xmu(prev_layer.dimensions());

      xmu.view() = *prev_layer - *broadcast_mean;

      // Step 9
      // dbeta = sum(dout, reduced by all dims except channel)
      Device_& device(dbeta.get_device());

      dbeta.view() = dout->sum(reduction_dims);

      // Step 8
      // dgamma = sum (dout * y, reduced by all dims except channel)
      DeviceTensor<Scalar, Rank, Device_, Layout> gamma_broad(broadcast_as_last_dim(gamma, broadcast_dims));
      DeviceTensor<Scalar, Rank, Device_, Layout> dxhat(dout.dimensions());
      dxhat.view() = *dout * *gamma_broad;
      dgamma.view() = (dout * xhat)->sum(reduction_dims);

      // Step 7
      // d_inv_std
      //TensorSingleDim d_inv_std = (dxhat * xmu).sum(reduction_dims);
      DeviceTensor<Scalar, Rank, Device_, Layout> dxmu1(dxhat.dimensions());
      dxmu1.view() = *dxhat * (1. / (*broadcast_var + eps).sqrt());

      // Step 6
      // TensorSingleDim d_std = -d_inv_std / (running_var + eps);

      // Step 5
      DeviceTensor<Scalar, 1, Device_> d_var(var.dimensions());
      d_var.view() = -0.5 * (dxhat * xmu)->sum(reduction_dims) / (*var + eps).pow(3. / 2.);

      // Step 4
      DeviceTensor<Scalar, Rank, Device_, Layout> d_var_broadcast = broadcast_as_last_dim(d_var, broadcast_dims);
      DeviceTensor<Scalar, Rank, Device_, Layout> d_sq(dout.dimensions());

      d_sq.view() = 1. / total_channel * *d_var_broadcast;

      // Step 3
      DeviceTensor<Scalar, Rank, Device_, Layout> dxmu2(d_sq.dimensions());
      dxmu2.view() = 2 * *xmu * *d_sq;

      // step 2
      DeviceTensor<Scalar, Rank, Device_, Layout> dx1(dxmu1.dimensions());
      dx1.view() = *dxmu1 + *dxmu2;

      DeviceTensor<Scalar, 1, Device_> dmu(dout->dimension(1));
      dmu.view() = -dx1->sum(reduction_dims);

      // step 1
      DeviceTensor<Scalar, Rank, Device_, Layout> dx2(dout.dimensions());
      DeviceTensor<Scalar, Rank, Device_, Layout> dmu_broadcast = broadcast_as_last_dim(dmu, broadcast_dims);

      dx2.view() = 1. / total_channel * *dmu_broadcast;

      // step 0
      layer_gradient.view() = *dx1 + *dx2;
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

    void set_weights(const DeviceTensor<Scalar, 1, Device_, Layout>& v) {
      gamma = v;
    }

    void set_bias(const DeviceTensor<Scalar, 1, Device_, Layout>& v) {
      beta = v;
    }

  private:
    DeviceTensor<Scalar, Rank, Device_> layer_output, layer_gradient, xhat;

    DeviceTensor<Scalar, 1, Device_> gamma, beta, running_mean, running_variance, mu, var;
    DeviceTensor<Scalar, 1, Device_> dbeta, dgamma;
    float momentum, eps;
    bool is_training;

  };

}