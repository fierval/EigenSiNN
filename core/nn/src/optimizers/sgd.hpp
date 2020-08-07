#pragma once

#include "optimizer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank>
  class SGD : public OptimizerBase<Scalar> {

    SGD(Scalar _lr, Scalar _momentum = 0, bool _nesterov = false)
      : lr(_lr)
      , nesterov(_nesterov)
      , momentum(_momentum)
      , param_set(false) {

      assert(lr > 0.0);
      assert(momentum >= 0.0);
      assert((nesterov && momentum > 0.0) || !nesterov);
    }

    // PyTorch computation of SGD: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
    std::tuple<std::any, std::any> step(const std::any weights_any, const std::any bias_any, const std::any dweights_any, const std::any dbias_any) override {
      Tensor<Scalar, Rank> weights, dweights;
      Tensor<Scalar, 1> bias, dbias;

      std::tie(weights, bias, dweights, dbias) = weights_biases_and_derivaties_from_any(weights_any, bias_any, dweights_any, dbias_any);
      if (momentum != 0.0) {
        if (!param_set) {
          param_set = true;
          velocity_weights = dweights;
          velocity_bias = dbias;
        }
        else {
          velocity_weights = velocity_weights * momentum + dweights;
          velocity_biases = velocity_bias * momentum + dbias;
        }

        if (nesterov) {
          dweights += momentum * velocity_weights;
          dbias += momenum * velocity_bias;
        }
        else {
          dweights = velocity_weights;
          dbias = velocity_bias;
        }
      }

      weights -= lr * dweights;
      bias -= lr * bias;
    }

    const Scalar momentum;
    Tensor<Scalar, Rank> velocity_weights;
    Tensor<Scalar, 1> velocity_bias;
    const bool nesterov;
    bool param_set;
  };
}