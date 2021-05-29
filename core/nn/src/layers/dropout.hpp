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

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class Dropout : public LayerBase<Scalar, Device_> {
  public:

    Dropout(float _prob = 0.5)
      : prob(_prob)
      , is_training(true)
      , inited(false) {
    }

    void init(const DeviceTensor<Scalar, Rank, Device_, Layout>& x)  {

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

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer.get_output());

      if (!inited) {
        inited = true;
        init(x);
      }

      // generate random uniform values
      rands.setRandom<internal::UniformRandomGenerator<float>>();
            
      // create condition
      if_tensor.view() = *rands >= *prob_tensor;
      mask.view() = if_tensor->select(*then_tensor, *else_tensor);

      layer_output.view() = *mask * *x;
    }

    // for derivations
    void backward(LayerBase<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> next_layer_grad(next_layer_grad_any);

      if (!is_training) { return; }

      layer_gradient.view() = *mask * *next_layer_grad;
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() {
      return layer_gradient.raw();
    }

    void set_training(bool _is_training) { 
      is_training = _is_training;
    }

    const bool get_training() {
      return is_training;
    }

    const std::string add_onnx_node(EigenModel<Scalar>& model, const std::string& input_name) {

      // Dropout is a noop during inference
      if (model.is_inference()) {
        return;
      }

      // 1. Add initializers
      prob_tensor.save_onnx_initializer(model);
      training_mode.save_onnx_initializer(model);

      // Dropout spec saves everything as input
      // So we need to wrap scalar values in tensors
      DeviceTensor<float, 0> prob_tensor;
      prob_tensor.setConstant(prob);

      DeviceTensor<bool, 0> training_mode;
      training_mode.setConstant(true);

      // 2. Add inputs
      auto* node = model.add_graph_node(op_type, {input_name, 
        prob_tensor.get_onnx_input_name(), training_mode.get_onnx_input_name() });

      const std::string out_name = node->output().Get(0);
      return out_name;
    }

  private:
    DeviceTensor<Scalar, Rank, Device_, Layout> mask;
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_gradient;
    DeviceTensor<bool, Rank, Device_, Layout> if_tensor;
    DeviceTensor<float, Rank, Device_, Layout> rands, then_tensor, else_tensor, prob_tensor;
    bool is_training, inited;
    const float prob;

    // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#Dropout
    static constexpr char op_type[] = "Dropout";

  };

}