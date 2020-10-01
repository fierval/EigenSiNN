#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>
#include <cstdlib>
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>
#include <limits>

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Dropout : public LayerBase<Scalar, Device_> {
  public:

    Dropout(float _prob = 0.5, Dispatcher<Device_>& _device =  LayerBase::default_dispatcher)
      : LayerBase(_device)
      , prob(_prob)
      , is_training(true)
      , inited(false) {
    }

    void init(const Tensor<Scalar, Rank>& x)  {

      using std::begin;
      using std::end;

      layer_gradient.resize(x.dimensions());
      layer_output.resize(x.dimensions());
      mask.resize(x.dimensions());
      rands.resize(x.dimensions());

      if_tensor.resize(x.dimensions());
      
      then_tensor.resize(x.dimensions());
      then_tensor.setConstant(1. / (1. - prob));

      else_tensor.resize(x.dimensions());
      else_tensor.setZero();

      prob_tensor.resize(x.dimensions());
      prob_tensor.setConstant(prob);
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer) override {
      
      if (!is_training) { return; }

      if (are_dims_unset(prev_layer.get_out_dims())) {
        set_dims(prev_layer.get_out_dims(), prev_layer.get_out_dims());
      }
      
      TensorMap<Tensor<Scalar, Rank>> x(prev_layer.get_output(), vector2array<int, Rank>(in_dims));

      if (!inited) {
        inited = true;
        init(x);
      }

      // generate random uniform values
      rands.setRandom<internal::UniformRandomGenerator<float>>();
            
      // create condition
      if_tensor.device(dispatcher.get_device()) = rands >= prob_tensor;
      mask.device(dispatcher.get_device()) = if_tensor.select(then_tensor, else_tensor);

      layer_output.device(dispatcher.get_device()) = mask * x;
    }

    // for derivations
    void backward(LayerBase<Scalar, Device_>& prev_layer_any, LayerBase<Scalar, Device_>& next_layer_grad_any) override {

      TensorMap<Tensor<Scalar, Rank>> next_layer_grad(next_layer_grad_any.get_output(), vector2array<int, Rank>(out_dims));

      if (!is_training) { return; }

      layer_gradient.device(dispatcher.get_device()) = mask * next_layer_grad;
    }

    Scalar * get_output() override {
      return layer_output.data();
    }

    Scalar * get_loss_by_input_derivative() {
      return layer_gradient.data();
    }

    void set_training(bool _is_training) { 
      is_training = _is_training;
    }

    const bool get_training() {
      return is_training;
    }

  private:
    Tensor<Scalar, Rank> mask;
    Tensor<Scalar, Rank> layer_output, layer_gradient;
    Tensor<bool, Rank> if_tensor;
    Tensor<float, Rank> rands, then_tensor, else_tensor, prob_tensor;
    bool is_training, inited;
    const float prob;
  };

}