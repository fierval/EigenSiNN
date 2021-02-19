#pragma once

#include "optimizer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  class SGD : public OptimizerBase<Scalar, Layout, Device_> {

  public:
    SGD(Scalar _lr, Scalar _momentum = 0, bool _nesterov = false)
      : OptimizerBase(_lr)
      , nesterov(_nesterov)
      , momentum(_momentum)
      {

      assert(lr > 0.0);
      assert(momentum >= 0.0);
      assert((nesterov && momentum > 0.0) || !nesterov);
    }

    // PyTorch computation of SGD: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
    inline DeviceWeightBiasTuple step(LayerBase<Scalar>& layer) override {
      
      DeviceTensor<Device_, Scalar, Rank, Layout> weights(layer.get_weights()), dweights(layer.get_loss_by_weights_derivative());
      DeviceTensor<Device_, Scalar, 1, Layout> bias(layer.get_bias()), dbias(layer.get_loss_by_bias_derivative());

      if (momentum != 0.0) {
        if (!param_set) {
          param_set = true;
          velocity_weights = dweights;
          velocity_bias = dbias;
        }
        else {
          velocity_weights = momentum * velocity_weights + dweights;
          velocity_bias = momentum * velocity_bias + dbias;
        }

        if (nesterov) {
          dweights += momentum * velocity_weights;
          dbias += momentum * velocity_bias;
        }
        else {
          dweights = velocity_weights;
          dbias = velocity_bias;
        }
      }

      weights -= lr * dweights;
      bias -= lr * dbias;
      return std::make_tuple(weights, bias);
    }

  private:
    const Scalar momentum;
    DeviceTensor<Device_, Scalar, Rank, Layout> velocity_weights;
    DeviceTensor<Device_, Scalar, 1, Layout> velocity_bias;
    const bool nesterov;
  };
}