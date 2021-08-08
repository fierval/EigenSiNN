#pragma once

#include <layers/layer_base.hpp>
#include <ops/conversions.hpp>
#include <onnx/op_defs.h>

using namespace  Eigen;

namespace EigenSinn {

  // ditching the Rank for this op
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class Add : public LayerBase<Scalar, Device_> {

  public:

    typedef std::unordered_map<std::string, PtrTensorAdaptor<Scalar, Device_>> LayerTensorAdapterMap;

    Add() : LayerBase<Scalar, Device_>(OnnxOpNames::add_op) { }

    void forward(LayerTensorAdapterMap& inputs) override {

      assert(inputs.size() == 2);

      // TODO: implement broadcasting?
      auto it = inputs.begin();
      PtrTensorAdaptor<Scalar, Device_> input1 = it->second;
      PtrTensorAdaptor<Scalar, Device_> input2 = (it + 1)->second;

      int rank = input1->get_dims().size();
      switch (rank) {
      case 2:
        DeviceTensor<Scalar, 2, Device_, Layout> o_2(input1->get_dims()), i1_2(input1), i2_2(input2);
        o_2.view() = *i1_2 + *i2_2;
        output = o_2.raw();
        break;
      case 3:
        DeviceTensor<Scalar, 3, Device_, Layout> o_3(input1->get_dims()), i1_3(input1), i2_3(input2);
        o_3.view() = *i1_3 + *i2_3;
        output = o_2.raw();
        break;
      case 4:
        DeviceTensor<Scalar, 4, Device_, Layout> o_4(input1->get_dims()), i1_4(input1), i2_4(input2);
        o_4.view() = *i1_4 + *i2_4;
        output = o_4.raw();
        break;
      default:
        assert(false);
        throw std::logic_error("not implemented for this rank");
      }
    }

    void backward(LayerTensorAdapterMap& prev_layer, PtrTensorAdaptor<Scalar, Device_>& next_layer_grad_any) override {

      // TODO: implement backprop with broadcasting, where next_layer_grad_any should be resized based on broadcasting

      for (auto& kv : prev_layer) {
        if (dinput.count(kv.first) == 0) {
          dinput.insert(std::make_pair(kv.first, next_layer_grad_any));
        }
        else {
          dinput[kv.first] = next_layer_grad_any;
        }
      }
    }

    void forward(PtrTensorAdaptor<Scalar, Device_>& inp) override {

      static_assert("Add::forward with only one argument!!");
    }

    PtrTensorAdaptor<Scalar, Device_> get_output() override {

      return output;
    }

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative() {
      static_assert("Add: use get_loss_by_input_derivative with layer name");
      return PtrTensorAdaptor<Scalar, Device_>();
    }

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative(std::string& layer_name) override {
      return dinput[layer_name];
    }

  private:
    PtrTensorAdaptor<Scalar, Device_> output;
    LayerTensorAdapterMap dinput;
  };

}