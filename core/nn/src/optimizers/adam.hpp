#pragma once

#include "optimizer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Adam : public OptimizerBase<Scalar, Device_> {

  public:
    Adam(Scalar _lr, Dispatcher<Device_>& _device = OptimizerBase::default_dispatcher, float _beta1 = 0.9, float _beta2 = 0.999, float _eps = 1e-8 )
      : OptimizerBase(_lr, _device)
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
    std::tuple<Scalar *, Scalar *> step(LayerBase<Scalar>& layer) override {

      Scalar* weights_any, * dweights_any, * bias_any, *dbias_any;
      array<Index, Rank> dims = layer

      Tensor<Scalar, Rank> weights, dweights;
      Tensor<Scalar, 1> bias, dbias;

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
      velocity_weights = beta2 * velocity_weights + (1 - beta2) * dweights.pow(2.);

      momentum_bias = beta1 * momentum_bias + (1 - beta1) * dbias;
      velocity_bias = beta2 * velocity_bias + (1 - beta2) * dbias.pow(2.);

      cur_beta1 *= beta1;
      cur_beta2 *= beta2;

      Tensor<Scalar, Rank> denom_weights = velocity_weights.sqrt() / sqrt(1 - cur_beta2) + eps;
      Tensor<Scalar, 1> denom_bias = velocity_bias.sqrt() / sqrt(1 - cur_beta2) + eps;

      Scalar step_size = lr / (1 - cur_beta1);

      weights -= step_size * momentum_weights / denom_weights;
      bias  -= step_size * momentum_bias / denom_bias;

      return make_tuple(std::make_any<Tensor<Scalar, Rank>>(weights), std::make_any<Tensor<Scalar, 1>>(bias));
    }

  private:
    const Scalar beta1, beta2, eps;
    Scalar cur_beta1, cur_beta2;
    Tensor<Scalar, Rank> velocity_weights, momentum_weights;
    Tensor<Scalar, 1> velocity_bias, momentum_bias;
  };
}