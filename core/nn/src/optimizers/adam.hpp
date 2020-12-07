#pragma once

#include "optimizer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = DefaultDevice>
  class Adam : public OptimizerBase<Scalar, Rank, Layout, Device_> {

  public:
    Adam(Scalar _lr, float _beta1 = 0.9, float _beta2 = 0.999, float _eps = 1e-8)
      : OptimizerBase(_lr)
      , beta1(_beta1)
      , beta2(_beta2)
      , eps(_eps)
      , cur_beta1(1)
      , cur_beta2(1)
       {
      assert(beta1 > 0.0);
      assert(beta2 > 0.0);
    }

    // Computation: https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
    DeviceWeightBiasTuple step(LayerBase<Scalar>& layer) override {

     DeviceTensor<Device_, Scalar, Rank, Layout> weights(layer.get_weights()), dweights(layer.get_loss_by_weights_derivative());
     DeviceTensor<Device_, Scalar, 1, Layout> bias(layer.get_bias()), dbias(layer.get_loss_by_bias_derivative());

      if (!param_set) {
        param_set = true;

        velocity_weights.resize(weights.dimensions());
        momentum_weights.resize(weights.dimensions());

        velocity_bias.resize(bias.dimensions());
        momentum_bias.resize(bias.dimensions());

        velocity_weights.setZero();
        momentum_weights.setZero();

        velocity_bias.setZero();
        momentum_bias.setZero();
      }

      // compute Mt and Vt
      momentum_weights = beta1 * momentum_weights + (1 - beta1) * dweights;
      velocity_weights.view() = beta2 * *velocity_weights + (1 - beta2) * dweights->pow(2.);

      momentum_bias = beta1 * momentum_bias + (1 - beta1) * dbias;
      velocity_bias.view() = beta2 * *velocity_bias + (1 - beta2) * dbias->pow(2.);

      cur_beta1 *= beta1;
      cur_beta2 *= beta2;

      DeviceTensor<Device_, Scalar, Rank, Layout> denom_weights(velocity_weights.dimensions());
      denom_weights.view() = velocity_weights->sqrt() / sqrt(1 - cur_beta2) + eps;

      DeviceTensor<Device_, Scalar, 1, Layout> denom_bias(velocity_bias.dimensions());
      denom_bias.view() = velocity_bias->sqrt() / sqrt(1 - cur_beta2) + eps;

      Scalar step_size = lr / (1 - cur_beta1);

      weights -= step_size * momentum_weights / denom_weights;
      bias -= step_size * momentum_bias / denom_bias;

      return std::make_tuple(weights, bias);
    }

  private:
    const Scalar beta1, beta2, eps;
    Scalar cur_beta1, cur_beta2;
    DeviceTensor<Device_, Scalar, Rank, Layout> velocity_weights, momentum_weights;
    DeviceTensor<Device_, Scalar, 1, Layout> velocity_bias, momentum_bias;
  };
}