#pragma once

#include <layers/layer_base.hpp>
#include <ops/conversions.hpp>
#include <onnx/op_defs.h>

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class Add : public LayerBase<Scalar, Device_> {

  public:
    Add() : LayerBase<Scalar, Device_>(OnnxOpNames::add_op) { }
  
    void forward(std::vector<PtrTensorAdapter<Scalar, Device_>>& inputs) override {

      assert(inputs.size() == 2);

      // TODO: implement broadcasting?
      DeviceTensor<Scalar, Rank, Device_, Layout> input1(inputs[0]);
      DeviceTensor<Scalar, Rank, Device_, Layout> input2(inputs[1]);

      if (output.size() == 0) {
        output.resize(input1.dimensions());
      }

      output.view() = *input1 + *input2;
    }

    void backward(std::vector<PtrTensorAdapter<Scalar, Device_>>& prev_layer, PtrTensorAdapter<Scalar, Device_>& next_layer_grad_any) override {
    
      // TODO: implement backprop with broadcasting, where next_layer_grad_any should be resized based on broadcasting

      DeviceTensor<Scalar, Rank, Device_, Layout> next_layer_grad(next_layer_grad_any);

      if (dinput.empty()) {
        dinput.resize(2);
      }

      std::fill_n(dinput.begin(), 2, next_layer_grad);
    }

    void forward(PtrTensorAdapter<Scalar, Device_>& inp) override {
      
      static_assert("Add::forward with only one argument!!");
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
    
      return output;
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() {
      static_assert("Add: use get_loss_by_input_derivatives (plural)!");
      return PtrTensorAdapter<Scalar, Device_>();
    }

    std::vector<PtrTensorAdapter<Scalar, Device_>> get_loss_by_input_derivatives() {
      

      return dinput;
    }

  private:
    DeviceTensor<Scalar, Rank, Device_, Layout> output;
    std::vector<DeviceTensor<Scalar, Rank, Device_, Layout>> dinput;
  };

}