#pragma once

#include "optimizer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class SGD : public OptimizerBase<Scalar, Device_> {

  public:
    SGD(Scalar _lr, Scalar _momentum = 0, bool _nesterov = false, Dispatcher<Device_>& _device = OptimizerBase::default_dispatcher)
      : OptimizerBase(_lr, _device)
      , nesterov(_nesterov)
      , momentum(_momentum)
      {

      assert(lr > 0.0);
      assert(momentum >= 0.0);
      assert((nesterov && momentum > 0.0) || !nesterov);
    }

    // PyTorch computation of SGD: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
    std::tuple<Scalar*, Scalar*> step(LayerBase<Scalar, Device_>& layer) override {
      
      array<Index, Rank> dims = vector2array< Rank>(layer.get_weight_dims());
      array<Index, 1> dims_bias = vector2array< 1>(layer.get_bias_dims());

      TensorMap<Tensor<Scalar, Rank>> weights(layer.get_weights(), dims), dweights(layer.get_loss_by_weights_derivative(), dims);
      TensorMap<Tensor<Scalar, 1>> bias(layer.get_bias(), dims_bias), dbias(layer.get_loss_by_bias_derivative(), dims_bias);

      if (momentum != 0.0) {
        if (!param_set) {
          param_set = true;
          velocity_weights = dweights;
          velocity_bias = dbias;
        }
        else {
          velocity_weights = velocity_weights * momentum + dweights;
          velocity_bias = velocity_bias * momentum + dbias;
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
      return std::make_tuple(weights.data(), bias.data());
    }

  private:
    const Scalar momentum;
    Tensor<Scalar, Rank> velocity_weights;
    Tensor<Scalar, 1> velocity_bias;
    const bool nesterov;
  };
}