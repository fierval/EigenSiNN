#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Softmax : public LayerBase<Device_> {
  public:
    Softmax(const Device_& _device = DefaultDevice()) :
      LayerBase(_device)
      , inited(false)
    {
      for (Index i = 0; i < Rank - 1; i++) {
          reduction_axes[i] = i + 1;
          reshape_dims[i] = 1;
      }

      reshape_dims[Rank - 1] = 1;
    }

    void forward(std::any prev_layer_any) override {
      Tensor<Scalar, Rank> prev_layer = from_any<Scalar, Rank>(prev_layer_any);
      
      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

      // reliable softmax
      Tensor<Scalar, 0> layer_max;
      layer_max.device(device) = prev_layer.maximum();

      Tensor<Scalar, Rank> layer_max_broadcast(prev_layer.dimensions());
      layer_max_broadcast.setConstant(layer_max(0));

      exp_all.device(device) = (prev_layer - layer_max_broadcast).exp();

      exp_sum.device(device) = exp_all.sum(reduction_axes);
      exp_sum_broadcast.device(device) = exp_sum.reshape(reshape_dims).broadcast(broadcast_dims);

      layer_output.device(device) = exp_all / exp_sum_broadcast;
    }

    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {
      auto dout = from_any<Scalar, Rank>(next_layer_grad_any);

      Tensor<Scalar, Rank> d_mul_exp(dims);
      d_mul_exp.device(device) = dout / exp_sum_broadcast;

      Tensor<Scalar, 1> d_mul_inv_x(dims[0]);
      d_mul_inv_x.device(device) = (exp_all * dout).sum(reduction_axes);

      Tensor<Scalar, 1> d_inv(dims[0]);
      d_inv.device(device) = -1. / exp_sum.pow(2) * d_mul_inv_x;

      Tensor<Scalar, Rank> d_sum_exp(dims);
      d_sum_exp.device(device) = d_inv.reshape(reshape_dims).broadcast(broadcast_dims);

      Tensor<Scalar, Rank> d_exp(dims);
      d_exp.device(device) = d_mul_exp + d_sum_exp;

      layer_grad.device(device) = exp_all * d_exp;
    }

    std::any get_output() override {
      return layer_output;
    };

    std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };

  private:
    void init_cached(Eigen::Tensor<Scalar, Rank>& prev_layer)
    {
      layer_output.resize(prev_layer.dimensions());
      layer_grad.resize(prev_layer.dimensions());
      exp_all.resize(prev_layer.dimensions());
      exp_sum.resize(prev_layer.dimension(0));
      exp_sum_broadcast.resize(prev_layer.dimensions());

      // broadcast the exp sum for future ops
      broadcast_dims = prev_layer.dimensions();
      broadcast_dims[0] = 1;

      reshape_dims[0] = prev_layer.dimension(0);
      dims = prev_layer.dimensions();
    }

    bool inited;
    Tensor<Scalar, Rank> layer_output, layer_grad, exp_all, exp_sum_broadcast;
    Tensor<Scalar, 1> exp_sum;
    array<Index, Rank - 1> reduction_axes;
    array<Index, Rank> broadcast_dims, reshape_dims, dims;
  };


}