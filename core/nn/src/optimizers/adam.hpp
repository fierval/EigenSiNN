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

      array<Index, Rank> dims = vector2array<int, Rank>(layer.get_weight_dims());
      array<Index, 1> dims_bias = vector2array<int, 1>(layer.get_bias_dims());

      TensorMap<Tensor<Scalar, Rank>> weights(layer.get_output(), dims), dweights(layer.get_loss_by_weights_derivative(), dims);
      TensorMap<Tensor<Scalar, 1>> bias(layer.get_bias(), dims_bias), dbias(layer.get_loss_by_bias_derivative(), dims_bias);

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
      momentum_weights.device(dispatcher.get_device()) = beta1 * momentum_weights + (1 - beta1) * dweights;
      velocity_weights.device(dispatcher.get_device()) = beta2 * velocity_weights + (1 - beta2) * dweights.pow(2.);

      momentum_bias.device(dispatcher.get_device()) = beta1 * momentum_bias + (1 - beta1) * dbias;
      velocity_bias.device(dispatcher.get_device()) = beta2 * velocity_bias + (1 - beta2) * dbias.pow(2.);

      cur_beta1.device(dispatcher.get_device()) *= beta1;
      cur_beta2.device(dispatcher.get_device()) *= beta2;

      Tensor<Scalar, Rank> denom_weights(velocity_weights.dimensions());
      denom_weights.device(dispatcher.get_device()) = velocity_weights.sqrt() / sqrt(1 - cur_beta2) + eps;

      Tensor<Scalar, 1> denom_bias(velocity_bias.dimensions());
      denom_bias.device(dispatcher.get_device()) = velocity_bias.sqrt() / sqrt(1 - cur_beta2) + eps;

      Scalar step_size = lr / (1 - cur_beta1);

      weights.device(dispatcher.get_device()) -= step_size * momentum_weights / denom_weights;
      bias.device(dispatcher.get_device()) -= step_size * momentum_bias / denom_bias;

      return make_tuple(weights.data(), bias.data());
    }

  private:
    const Scalar beta1, beta2, eps;
    Scalar cur_beta1, cur_beta2;
    Tensor<Scalar, Rank> velocity_weights, momentum_weights;
    Tensor<Scalar, 1> velocity_bias, momentum_bias;
  };
}