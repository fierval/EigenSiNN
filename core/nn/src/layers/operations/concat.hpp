#pragma once

#include <layers/layer_base.hpp>
#include <ops/conversions.hpp>
#include <onnx/op_defs.h>

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class Concat : public LayerBase<Scalar, Device_> {

  public:

    typedef std::unordered_map<std::string, PtrTensorAdapter<Scalar, Device_>> LayerTensorAdapterMap;

    Concat(int _axis) 
      : LayerBase<Scalar, Device_>(OnnxOpNames::concat_op) 
      , axis(_axis) {
    
      }

    void forward(LayerTensorAdapterMap& inputs) override {

      assert(inputs.size() == 2);

      // TODO: implement broadcasting?
      auto it = inputs.begin();
      DeviceTensor<Scalar, Rank, Device_, Layout> input1(it->second);
      DeviceTensor<Scalar, Rank, Device_, Layout> input2((it + 1)->second);

      if (output.size() == 0) {
        output.resize(input1.dimensions());
      }

      output.view() = *input1 + *input2;
    }

    void backward(LayerTensorAdapterMap& prev_layer, PtrTensorAdapter<Scalar, Device_>& next_layer_grad_any) override {

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

    void forward(PtrTensorAdapter<Scalar, Device_>& inp) override {

      static_assert("Concat::forward with only one argument!!");
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {

      return output;
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() {
      static_assert("Concat: use get_loss_by_input_derivative with a specific layer name!");
      return PtrTensorAdapter<Scalar, Device_>();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative(std::string& layer_name) override {
      return dinput[layer_name];
    }

  private:
    DeviceTensor<Scalar, Rank, Device_, Layout> output;
    LayerTensorAdapterMap dinput;
    int axis;
  };

}