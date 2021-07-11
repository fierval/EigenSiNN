#pragma once

#include "optimizer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class Adam : public OptimizerBase<Scalar, Device_, Layout> {

  public:
    Adam(float _lr, float _beta1 = 0.9, float _beta2 = 0.999, float _eps = 1e-8)
      : OptimizerBase<Scalar, Device_, Layout>(_lr)
      , beta1(_beta1)
      , beta2(_beta2)
      , eps(_eps)
      , cur_beta1(1)
      , cur_beta2(1) {

      assert(beta1 > 0.0);
      assert(beta2 > 0.0);
    }

    Adam(Adam& a)
      : OptimizerBase<Scalar, Device_, Layout>(a.lr)
      , beta1(a.beta1)
      , beta2(a.beta2)
      , eps(a.eps) {

      if (this == &a) {
        return;
      }

      cur_beta1 = 1;
      cur_beta2 = 1;
    }

    // Computation: https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
    inline DeviceWeightBiasTuple<Scalar, Device_> step(LayerBase<Scalar, Device_>& layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> weights(layer.get_weights()), dweights(layer.get_loss_by_weights_derivative());
      DeviceTensor<Scalar, 1, Device_, Layout> bias(layer.get_bias()), dbias(layer.get_loss_by_bias_derivative());

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
      momentum_weights.view() = beta1 * *momentum_weights + (1 - beta1) * *dweights;
      velocity_weights.view() = beta2 * *velocity_weights + (1 - beta2) * dweights->pow(2.);

      momentum_bias.view() = beta1 * *momentum_bias + (1 - beta1) * *dbias;
      velocity_bias.view() = beta2 * *velocity_bias + (1 - beta2) * dbias->pow(2.);

      cur_beta1 *= beta1;
      cur_beta2 *= beta2;

      DeviceTensor<Scalar, Rank, Device_, Layout> denom_weights(velocity_weights.dimensions());
      denom_weights.view() = velocity_weights->sqrt() / sqrt(1 - cur_beta2) + eps;

      DeviceTensor<Scalar, 1, Device_, Layout> denom_bias(velocity_bias.dimensions());
      denom_bias.view() = velocity_bias->sqrt() / sqrt(1 - cur_beta2) + eps;

      Scalar step_size = lr / (1 - cur_beta1);

      weights.view() = *weights - step_size * *momentum_weights / *denom_weights;
      bias.view() = *bias - step_size * *momentum_bias / *denom_bias;

      return std::make_tuple(weights.raw(), bias.raw());
    }

  private:
    const Scalar beta1, beta2, eps;
    Scalar cur_beta1, cur_beta2;
    DeviceTensor<Scalar, Rank, Device_, Layout> velocity_weights, momentum_weights;
    DeviceTensor<Scalar, 1, Device_, Layout> velocity_bias, momentum_bias;
  };
}