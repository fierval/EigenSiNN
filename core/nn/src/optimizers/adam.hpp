#pragma once

#include "optimizer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank>
  class Adam : public OptimizerBase<Scalar> {

  public:
    Adam(Scalar _lr, Scalar _beta1 = 0.9, Scalar _beta2 = 0.99, Scalar _eps = 1e-8)
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
    std::tuple<std::any, std::any> step(const std::any weights_any, const std::any bias_any, const std::any dweights_any, const std::any dbias_any) override {
      Tensor<Scalar, Rank> weights, dweights;
      Tensor<Scalar, 1> bias, dbias;

      std::tie(weights, bias, dweights, dbias) = weights_biases_and_derivaties_from_any<Scalar, Rank>(weights_any, bias_any, dweights_any, dbias_any);

      if (!param_set) {
        param_set = true;

        velocity_weights.resize(weights.dimensions());
        momentum_weights.resize(weights.dimensions());

        velocity_weights.setZero();
        momentum_weights.setZero();

        veloicty_bias.setZero();
        momentum_bias.setZero();
      }

      // compute Mt and Vt
      momentum_weights = beta1 * momentum_weights + (1 - beta1) * dweights;
      velocity_weights = beta2 * velocity_weights + (1 - beta2) * dweights.pow(2.);

      momentum_bias = beta1 * momentum_bias + (1 - beta1) * dbias;
      velocity_bias = beta2 * velocity_bias + (1 - beta2) * dbias.pow(2.);

      cur_beta1 *= beta1;
      cur_beta2 *= beta2;

      Tensor<Scalar, Rank> momentum_hat_weights = momentum_weights / (1 - cur_beta1);
      Tensor<Scalar, Rank> velocity_hat_weights = velocity_weights / (1 - cur_beta2);

      Tensor<Scalar, 1> momentum_hat_bias = momentum_bias / (1 - cur_beta1);
      Tensor<Scalar, 1> velocity_hat_bias = velocity_bias / (1 - cur_beta2);

      Tensor<Scalar, Rank> step_size_weights = lr * momentum_weights.inverse();
      Tensor<Scalar, 1> step_size_bias = lr * momentum_bias.inverse();

      weights -= step_size_weights * momentum_hat_weights / (velocity_hat_weights.sqrt() + eps);
      bias  -= step_size_bias * momentum_hat_bias / (velocity_hat_bias.sqrt() + eps);

      return make_tuple(std::make_any<Tensor<Scalar, Rank>>(weights), std::make_any<Tensor<Scalar, 1>>(bias));
    }

  private:
    const Scalar beta1, beta2, eps;
    Scalar cur_beta1, cur_beta2;
    Tensor<Scalar, Rank> velocity_weights, momentum_weights;
    Tensor<Scalar, 1> velocity_bias, momentum_bias;
  };
}