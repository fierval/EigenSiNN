#pragma once

#include "optimizer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class SGD : public OptimizerBase<Scalar, Device_, Layout> {

  public:
    SGD(Scalar _lr, Scalar _momentum = 0, bool _nesterov = false)
      : OptimizerBase<Scalar, Device_, Layout>(_lr)
      , nesterov(_nesterov)
      , momentum(_momentum)
      {

      assert(lr > 0.0);
      assert(momentum >= 0.0);
      assert((nesterov && momentum > 0.0) || !nesterov);
    }

    // PyTorch computation of SGD: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
    inline DeviceWeightBiasTuple<Scalar, Device_> step(LayerBase<Scalar, Device_>& layer) override {
      
      DeviceTensor<Scalar, Rank, Device_, Layout> weights(layer.get_weights()), dweights(layer.get_loss_by_weights_derivative());
      DeviceTensor<Scalar, 1, Device_, Layout> bias(layer.get_bias()), dbias(layer.get_loss_by_bias_derivative());

      if (momentum != 0.0) {
        if (!param_set) {
          param_set = true;
          velocity_weights = dweights.clone();
          velocity_bias = dbias.clone();
        }
        else {
          velocity_weights.view() = momentum * *velocity_weights + *dweights;
          velocity_bias.view() = momentum * *velocity_bias + *dbias;
        }

        if (nesterov) {
          dweights.view() = *dweights + momentum * *velocity_weights;
          dbias.view() = *dbias + momentum * *velocity_bias;
        }
        else {
          dweights.view() = *velocity_weights;
          dbias.view() = *velocity_bias;
        }
      }

      weights.view() = *weights - lr * *dweights;
      bias.view() = *bias - lr * *dbias;
      return std::make_tuple(weights.raw(), bias.raw());
    }

  private:
    const Scalar momentum;
    DeviceTensor<Scalar, Rank, Device_, Layout> velocity_weights;
    DeviceTensor<Scalar, 1, Device_, Layout> velocity_bias;
    const bool nesterov;
  };
}