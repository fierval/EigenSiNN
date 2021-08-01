#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"
#include <onnx/op_defs.h>

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class Softmax : public LayerBase<Scalar, Device_> {
  public:
    Softmax() 
      : LayerBase<Scalar, Device_>(OnnxOpNames::softmax_op)
      , inited(false)
    {
      is_cudnn = Rank > 2 && Layout == RowMajor;

      for (Index i = 0; i < Rank - 1; i++) {
          reduction_axes[i] = i + 1;
          reshape_dims[i] = 1;
          ones_dims[i] = 1;
      }

      ones_dims[Rank - 1] = 1;
      reshape_dims[Rank - 1] = 1;
    }

    void forward(PtrTensorAdapter<Scalar, Device_>& prev_layer_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> prev_layer(prev_layer_any);
      
      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

      // reliable softmax
      DeviceTensor<Scalar, Rank, Device_, Layout> layer_max(prev_layer.dimensions());
      layer_max.view() = prev_layer->maximum().reshape(ones_dims).broadcast(dims);

      exp_all.view() = (*prev_layer - *layer_max).exp();

      exp_sum.view() = exp_all->sum(reduction_axes);
      exp_sum_broadcast.view() = exp_sum->reshape(reshape_dims).broadcast(broadcast_dims);

      layer_output.view() = *exp_all / *exp_sum_broadcast;
    }

    void backward(PtrTensorAdapter<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {
      
      DeviceTensor<Scalar, Rank, Device_, Layout> dout(next_layer_grad_any);

      DeviceTensor<Scalar, Rank, Device_, Layout> d_mul_exp(dims);
      d_mul_exp.view() = *dout / *exp_sum_broadcast;

      DeviceTensor<Scalar, 1, Device_, Layout> d_mul_inv_x(dims[0]);
      d_mul_inv_x.view() = (*exp_all * *dout).sum(reduction_axes);

      DeviceTensor<Scalar, 1, Device_, Layout> d_inv(dims[0]);
      d_inv.view() = -1. / exp_sum->pow(2) * *d_mul_inv_x;

      DeviceTensor<Scalar, Rank, Device_, Layout> d_sum_exp(dims);
      d_sum_exp.view() = d_inv->reshape(reshape_dims).broadcast(broadcast_dims);

      DeviceTensor<Scalar, Rank, Device_, Layout> d_exp(dims);
      d_exp.view() = *d_mul_exp + *d_sum_exp;

      layer_grad.view() = *exp_all * *d_exp;
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    };

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return layer_grad.raw();
    };

    // Save to ONNX
    // TODO: For the 2-dim softmax
    const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override {

      // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#Softmax
      auto* node = model.add_graph_node(get_layer_name(), get_op_name(), input_name);

      model.add_attr(node, "axis", 1);

      // not part of ONNX but necessary for loading
      model.add_attr(node, "rank", Rank);

      return node->output().Get(0);
    }

    const std::vector<Index> onnx_out_dims() override {
      return layer_output.vec_dims();
    }

  private:
    void init_cached(const DeviceTensor<Scalar, Rank, Device_, Layout>& prev_layer)
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
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_grad, exp_all, exp_sum_broadcast;
    DeviceTensor<Scalar, 1, Device_, Layout> exp_sum;
    array<Index, Rank - 1> reduction_axes;
    array<Index, Rank> broadcast_dims, reshape_dims, dims, ones_dims;
  };


}