#pragma once

#include "layer_base.hpp"
#include <ops/batchnorm.hpp>
#include <device/tensor_view.hpp>

using namespace  Eigen;
using std::unique_ptr;
using std::make_unique;

namespace EigenSinn {

  // NCHW format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
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

      beta.set_data_from_host(_beta);
      beta.set_data_from_host(_gamma);
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer_base) override {

      if (are_dims_unset(prev_layer_base.get_out_dims())) {

        DSizes<Index, Rank> dims = vector2Dsizes<Rank>(prev_layer_base.get_out_dims());

        set_dims(prev_layer_base.get_out_dims(), prev_layer_base.get_out_dims());
        layer_gradient = create_device_ptr<Device_, Scalar, Rank>(dims, device);
        layer_output = create_device_ptr<Device_, Scalar, Rank>(dims, device);
        xhat = create_device_ptr<Device_, Scalar, Rank>(dims, device);
      }

      TensorView<Scalar, Rank> prev_layer(prev_layer_base.get_output(), vector2array< Rank>(in_dims));

        std::tie(layer_output, xhat, running_mean, running_variance, mu, var) =
        batch_norm<Scalar, Rank, Device_>(prev_layer, gamma, beta, eps, momentum, running_mean, running_variance, is_training, device);
    }

    // see https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    // for derivations
    void backward(LayerBase<Scalar, Device_>& prev_layer_any, Scalar* next_layer_grad_any) override {

      TensorView<Scalar, Rank> prev_layer(prev_layer_any.get_output(), vector2array<Rank>(in_dims));
      TensorView<Scalar, Rank> dout(next_layer_grad_any, vector2array< Rank>(out_dims));

      DSizes<Index, Rank - 1> reduction_dims;
      DSizes<Index, Rank> broadcast_dims;

      float total_channel = 1.;
      for (int i = 0; i < Rank; i++) {
        if (i == (int)ImageDims::channel) {
          continue;
        }
        total_channel *= prev_layer.dimension(i);
      }

      std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims<Scalar, Rank>(dout);

      //broadcast values
      TensorView<Scalar, Rank> broadcast_mean = broadcast_as_last_dim<Scalar, Rank, Device_>(mu, broadcast_dims, device);
      TensorView<Scalar, Rank> broadcast_var = broadcast_as_last_dim<Scalar, Rank, Device_>(var, broadcast_dims, device);
      TensorView<Scalar, Rank> xmu = create_device_view<Device_, Scalar, Rank>(prev_layer.dimensions(), device);

      xmu.device(device) = (prev_layer - broadcast_mean);

      // Step 9
      // dbeta = sum(dout, reduced by all dims except channel)
      dbeta.device(device) = dout.sum(reduction_dims);

      // Step 8
      // dgamma = sum (dout * y, reduced by all dims except channel)
      TensorView<Scalar, Rank> gamma_broad(broadcast_as_last_dim<Scalar, Rank, Device_>(gamma, broadcast_dims, device));
      TensorView<Scalar, Rank> dxhat = create_device_view<Device_, Scalar, Rank>(dout.dimensions(), device);
      dxhat.device(device) = dout * gamma_broad;
      dgamma.device(device) = (dout * *xhat).sum(reduction_dims);

      // Step 7
      // d_inv_std
      //TensorSingleDim d_inv_std = (dxhat * xmu).sum(reduction_dims);
      TensorView<Scalar, Rank> dxmu1 = create_device_view<Device_, Scalar, Rank>(dxhat.dimensions(), device);
      dxmu1.device(device) = dxhat * (1. / (broadcast_var + eps).sqrt());

      // Step 6
      // TensorSingleDim d_std = -d_inv_std / (running_var + eps);

      // Step 5
      TensorView<Scalar, 1> d_var = create_device_view<Device_, Scalar, 1>(var.dimensions(), device);
      d_var.device(device) = -0.5 * (dxhat * xmu).sum(reduction_dims) / (var + eps).pow(3. / 2.);

      // Step 4
      TensorView<Scalar, Rank> d_var_broadcast = broadcast_as_last_dim<Scalar, Rank>(d_var, broadcast_dims, device);
      TensorView<Scalar, Rank> d_sq = create_device_view<Device_, Scalar, Rank>(dout.dimensions(), device);

      d_sq.device(device) = 1. / total_channel * d_var_broadcast;

      // Step 3
      TensorView<Scalar, Rank> dxmu2 = create_device_view<Device_, Scalar, Rank>(d_sq.dimensions(), device);
      dxmu2.device(device) = 2 * xmu * d_sq;

      // step 2
      TensorView<Scalar, Rank> dx1 = create_device_view<Device_, Scalar, Rank>(dxmu1.dimensions(), device);
      dx1.device(device) = dxmu1 + dxmu2;

      TensorView<Scalar, 1> dmu = create_device_view<Device_, Scalar, 1>(DSizes<Index, 1>{dout.dimension(1)}, device);
      dmu.device(device) = -dx1.sum(reduction_dims);

      // step 1
      TensorView<Scalar, Rank> dx2 = create_device_view<Device_, Scalar, Rank>(dout.dimensions(), device);
      TensorView<Scalar, Rank> dmu_broadcast = broadcast_as_last_dim<Scalar, Rank>(dmu, broadcast_dims, device);

      dx2.device(device) = 1. / total_channel * dmu_broadcast;

      // step 0
      layer_gradient->device(device) = dx1 + dx2;
    }

    Scalar* get_output() {

      return layer_output->data();
    }

    Scalar* get_loss_by_input_derivative() {
      return layer_gradient->data();
    }

    Scalar* get_loss_by_weights_derivative() override {
      return dgamma.data();
    }

    Scalar* get_loss_by_bias_derivative() override {
      return dbeta.data();
    }

    inline void SetTraining(bool training) {
      is_training = training;
    }

    inline bool IsTraining() {
      return is_training;
    }

    Scalar* get_weights() override {
      return gamma.data();
    }

    Scalar* get_bias() override {
      return beta.data();
    }

  private:
    DeviceTensor<Device_, Scalar, Rank> layer_output, layer_gradient, xhat;

    DeviceTensor<Device_, Scalar, 1> gamma, beta, running_mean, running_variance, mu, var;
    DeviceTensor<Device_, Scalar, 1> dbeta, dgamma;
    float momentum, eps;
    bool is_training;

  };

}