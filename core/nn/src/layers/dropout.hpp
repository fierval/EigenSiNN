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
  class Dropout : public LayerBase<Scalar> {
  public:

    Dropout(float _prob = 0.5)
      : prob(_prob)
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

    void forward(LayerBase<Scalar>& prev_layer) override {
      
      if (!is_training) { return; }

      DeviceTensor<Device_, Scalar, Rank> x(prev_layer.get_output());

      if (!inited) {
        inited = true;
        init(x);
      }

      // generate random uniform values
      rands.setRandom<internal::UniformRandomGenerator<float>>();
            
      // create condition
      if_tensor.view() = *rands >= *prob_tensor;
      mask.view() = if_tensor->select(*then_tensor, *else_tensor);

      layer_output = mask * x;
    }

    // for derivations
    void backward(LayerBase<Scalar>& prev_layer_any, std::any next_layer_grad_any) override {

      DeviceTensor<Device_, Scalar, Rank> next_layer_grad(next_layer_grad_any);

      if (!is_training) { return; }

      layer_gradient = mask * next_layer_grad;
    }

    std::any get_output() override {
      return layer_output;
    }

    std::any get_loss_by_input_derivative() {
      return layer_gradient;
    }

    void set_training(bool _is_training) { 
      is_training = _is_training;
    }

    const bool get_training() {
      return is_training;
    }

  private:
    DeviceTensor<Device_, Scalar, Rank> mask;
    DeviceTensor<Device_, Scalar, Rank> layer_output, layer_gradient;
    DeviceTensor<Device_, bool, Rank> if_tensor;
    DeviceTensor<Device_, float, Rank> rands, then_tensor, else_tensor, prob_tensor;
    bool is_training, inited;
    const float prob;
  };

}