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

    typedef std::unordered_map<std::string, PtrTensorAdaptor<Scalar, Device_>> LayerTensorAdaptorMap;
    typedef std::vector<PtrTensorAdaptor<Scalar, Device_>> TensorAdaptorVector;
    typedef std::vector<std::string> StringVector;

    Concat(int _axis)
      : LayerBase<Scalar, Device_>(OnnxOpNames::concat_op)
      , axis(_axis) {

    }

    void forward(LayerTensorAdaptorMap& inputs) override {

      auto inp_vector = get_inputs_from_map(inputs);
      auto inp_names = get_names_from_map(inputs);

      int rank = inp_vector[0]->get_dims().size();
      switch (rank) {
      case 2:
        exec_forward<2>(inp_vector, inp_names);
        break;
      case 3:
        exec_forward<3>(inp_vector, inp_names);
        break;
      case 4:
        exec_forward<4>(inp_vector, inp_names);
        break;
      default:
        assert(false);
        throw std::logic_error("not implemented for this rank");
      }
    }

    void backward(LayerTensorAdaptorMap& prev_layer, PtrTensorAdaptor<Scalar, Device_>& next_layer_grad_any) override {

      auto inp_vector = get_inputs_from_map(prev_layer);
      auto inp_names = get_names_from_map(prev_layer);

      int rank = inp_vector[0]->get_dims().size();

      switch (rank) {
      case 2:
        exec_backward<2>(next_layer_grad_any, names);
        break;
      case 3:
        exec_backward<3>(next_layer_grad_any, names);
        break;
      case 4:
        exec_backward<4>(next_layer_grad_any, names);
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

    void forward(PtrTensorAdaptor<Scalar, Device_>& inp) override {

      static_assert("Concat::forward with only one argument!!");
    }

    PtrTensorAdaptor<Scalar, Device_> get_output() override {

      return output;
    }

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative() {
      static_assert("Concat: use get_loss_by_input_derivative with a specific layer name!");
      return PtrTensorAdaptor<Scalar, Device_>();
    }

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative(std::string& layer_name) override {
      return dinput[layer_name];
    }

  private:
    // Functions to do the actual work, rank and layout dependent
    template<Index Rank, int Layout = RowMajor>
    void exec_forward(TensorAdaptorVector& inputs, StringVector& names) {

      auto in1_dims = inputs[0]->get_dims();
      auto in2_dims = inputs[1]->get_dims();
      auto out_dims(in1_dims.size());

      // output dimensions grow around the concatenation axis
      std::copy(in1_dims.begin(), in1_dims.end(), out_dims.begin());
      out_dims[axis] += in2_dims[axis];

      DeviceTensor<Scalar, Rank, Device_, Layout> out(out_dims), i1(inputs[0]), i2(inputs[1]);
      // this will throw if all dimensions (except axis) aren't equal
      out.view() = i1->concatenate(*i2, axis);
      output = out.raw();

      // store starting positions so we can correctly split on the backward step
      if (layer_to_starts.empty()) {

        std::vector<Index> sizes(rank);
        std::fill_n(sizes.begin(), rank, 0);

        layer_to_start.insert(names[0], sizes);
        layer_to_extents.insert(names[0], in1_dims);

        // the second layer starts where the first one ends
        layer_to_start.insert(names[1], in1_dims);
        layer_to_extents.insert(names[1], in2_dims);

      }
    }

    // we don't need the in-coming tensors for this, just their dimensions 
    // which we have saved
    template<Index Rank, int Layout = RowMajor>
    void exec_backward(PtrTensorAdaptor<Scalar, Device_>& next_layer_grad_any, StringVector& names) {

      DeviceTensor<Scalar, Rank, Device_, Layout> out(next_layer_grad_any);

      // the order of "names" vector is the order of in-coming layers in the prev_layer map
      DeviceTensor<Scalar, Rank, Device_, layout> dinput1(layer_to_extents[names[0]]), dinput2(layer_to_extents[names[1]]);
      dinput1.view() = out->slice(vec2dims<Rank>(layer_to_start[names[0]]), vec2dims<Rank>(layer_to_extents[names[0]]));
      dinput2.view() = out->slice(vec2dims<Rank>(layer_to_start[names[1]]), vec2dims<Rank>(layer_to_extents[names[1]]));

      dinput.clear();
      dinput.insert(names[0], dinput1);
      dinput.insert(names[1], dinput2);
    }

    PtrTensorAdaptor<Scalar, Device_> output;
    LayerTensorAdaptorMap dinput;
    int axis;
    std::unordered_map<std::string, std::vector<Index>> layer_to_starts, layer_to_extents;

  };

}