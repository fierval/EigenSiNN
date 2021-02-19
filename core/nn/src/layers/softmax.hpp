#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = ThreadPoolDevice>
  class Softmax : public LayerBase<Scalar> {
  public:
    Softmax() 
      : inited(false)
    {
      for (Index i = 0; i < Rank - 1; i++) {
          reduction_axes[i] = i + 1;
          reshape_dims[i] = 1;
          ones_dims[i] = 1;
      }

      ones_dims[Rank - 1] = 1;
      reshape_dims[Rank - 1] = 1;
    }

    void forward(LayerBase<Scalar>& prev_layer_any) override {

      DeviceTensor<Device_, Scalar, Rank, Layout> prev_layer(prev_layer_any.get_output());
      
      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

      // reliable softmax
      DeviceTensor<Device_, Scalar, Rank, Layout> layer_max(prev_layer.dimensions());
      layer_max.view() = prev_layer->maximum().reshape(ones_dims).broadcast(dims);

      exp_all.view() = (prev_layer - layer_max)->exp();

      exp_sum.view() = exp_all->sum(reduction_axes);
      exp_sum_broadcast.view() = exp_sum->reshape(reshape_dims).broadcast(broadcast_dims);

      layer_output = exp_all / exp_sum_broadcast;
    }

    void backward(LayerBase<Scalar>& prev_layer_any, std::any next_layer_grad_any) override {
      
      DeviceTensor<Device_, Scalar, Rank, Layout> dout(next_layer_grad_any);

      DeviceTensor<Device_, Scalar, Rank, Layout> d_mul_exp(dims);
      d_mul_exp = dout / exp_sum_broadcast;

      DeviceTensor<Device_, Scalar, 1, Layout> d_mul_inv_x(dims[0]);
      d_mul_inv_x.view() = (exp_all * dout)->sum(reduction_axes);

      DeviceTensor<Device_, Scalar, 1, Layout> d_inv(dims[0]);
      d_inv.view() = -1. / exp_sum->pow(2) * *d_mul_inv_x;

      DeviceTensor<Device_, Scalar, Rank, Layout> d_sum_exp(dims);
      d_sum_exp.view() = d_inv->reshape(reshape_dims).broadcast(broadcast_dims);

      DeviceTensor<Device_, Scalar, Rank, Layout> d_exp(dims);
      d_exp = d_mul_exp + d_sum_exp;

      layer_grad = exp_all * d_exp;
    }

    std::any get_output() override {
      return layer_output;
    };

    std::any get_loss_by_input_derivative() override {
      return layer_grad;
    };

  private:
    void init_cached(const DeviceTensor<Device_, Scalar, Rank, Layout>& prev_layer)
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
    DeviceTensor<Device_, Scalar, Rank, Layout> layer_output, layer_grad, exp_all, exp_sum_broadcast;
    DeviceTensor<Device_, Scalar, 1, Layout> exp_sum;
    array<Index, Rank - 1> reduction_axes;
    array<Index, Rank> broadcast_dims, reshape_dims, dims, ones_dims;
  };


}