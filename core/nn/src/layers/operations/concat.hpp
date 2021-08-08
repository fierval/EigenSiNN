#pragma once

#include <layers/layer_base.hpp>
#include <ops/conversions.hpp>
#include <onnx/op_defs.h>

using namespace  Eigen;

namespace EigenSinn {

  // ditching the rank here
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class Concat : public LayerBase<Scalar, Device_> {

  public:

    typedef std::unordered_map<std::string, PtrTensorAdapter<Scalar, Device_>> LayerTensorAdapterMap;

    Concat(int _axis) 
      : LayerBase<Scalar, Device_>(OnnxOpNames::concat_op) 
      , axis(_axis) {
    
      }

    void forward(LayerTensorAdapterMap& inputs) override {

      assert(inputs.size() == 2);

      assert(inputs.size() == 2);

      // TODO: implement broadcasting?
      auto it = inputs.begin();
      PtrTensorAdapter<Scalar, Device_> input1 = it->second;
      PtrTensorAdapter<Scalar, Device_> input2 = (it + 1)->second;

      int rank = input1->get_dims().size();
      switch (rank) {
      case 2:
        DeviceTensor<Scalar, 2, Device_, Layout> o_2(input1->get_dims()), i1_2(input1), i2_2(input2);
        o_2.view() = i1_2->concatenate(*i2_2, axis);
        output = o_2.raw();
        break;
      case 3:
        DeviceTensor<Scalar, 3, Device_, Layout> o_3(input1->get_dims()), i1_3(input1), i2_3(input2);
        o_3.view() = i1_3->concatenate(*i2_3, axis);
        output = o_2.raw();
        break;
      case 4:
        DeviceTensor<Scalar, 4, Device_, Layout> o_4(input1->get_dims()), i1_4(input1), i2_4(input2);
        o_4.view() = i1_4->concatenate(*i2_4, axis);
        output = o_4.raw();
        break;
      default:
        assert(false);
        throw std::logic_error("not implemented for this rank");
      }
    }

    void backward(LayerTensorAdapterMap& prev_layer, PtrTensorAdapter<Scalar, Device_>& next_layer_grad_any) override {

      auto it = prev_layer.begin();
      PtrTensorAdapter<Scalar, Device_> input1 = it->second;
      PtrTensorAdapter<Scalar, Device_> input2 = (it + 1)->second;

      std::string name1(it->first), name2((it + 1)->first);

      int rank = input1->get_dims().size();
      switch (rank) {
      case 2:
        DeviceTensor<Scalar, 2, Device_, Layout> o_2(next_layer_grad_any->get_dims()), i1_2(input1->get_dims()), i2_2(input1->get_dims());
        DSizes<Index, 2>
        break;
      case 3:
        DeviceTensor<Scalar, 3, Device_, Layout> o_3(next_layer_grad_any->get_dims()), i1_3(input1->get_dims()), i2_3(input1->get_dims());
        o_3.view() = i1_3->concatenate(*i2_3, axis);
        output = o_2.raw();
        break;
      case 4:
        DeviceTensor<Scalar, 4, Device_, Layout> o_4(next_layer_grad_any->get_dims()), i1_4(input1->get_dims()), i2_4(input1->get_dims());
        o_4.view() = i1_4->concatenate(*i2_4, axis);
        output = o_4.raw();
        break;
      default:
        assert(false);
        throw std::logic_error("not implemented for this rank");
      }

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
    PtrTensorAdapter<Scalar, Device_> output;
    LayerTensorAdapterMap dinput;
    int axis;
  };

}