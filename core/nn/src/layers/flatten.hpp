#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include <ops/conversions.hpp>

#include <onnx/op_defs.h>

namespace EigenSinn {

  // flatten convolution layer
  template<typename Scalar, typename Device_= ThreadPoolDevice, int Layout = RowMajor>
  class Flatten : public LayerBase<Scalar, Device_> {

  public:
    Flatten() : LayerBase<Scalar, Device_>(flatten_op) {
    
    }

    void forward(PtrTensorAdapter<Scalar, Device_>& prev_layer_any) {
      
      DeviceTensor<Scalar, 4, Device_, Layout> orig(prev_layer_any);

      original_dimensions = orig.dimensions();

      unfolded = unfold_kernel<Scalar>(orig);

    }

    void backward(PtrTensorAdapter<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) {

      DeviceTensor<Scalar, 2, Device_, Layout> unf_dout(next_layer_grad);

      folded = fold_kernel<Scalar>(unf_dout, original_dimensions);
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return unfolded.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return folded.raw();
    }

    const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override {

      // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#Flatten
      // 1. add ONNX node with its inputs, outputs, and names
      onnx::NodeProto* node = model.add_graph_node(flatten_op, input_name);
      // single output
      const std::string out_name = node->output().Get(0);

      // 2. create attributes. We are flattening all axes except 0
      auto axis_attr = node->add_attribute();
      axis_attr->set_name("axis");
      axis_attr->set_i(1);
      axis_attr->set_type(onnx::AttributeProto::AttributeType::AttributeProto_AttributeType_INT);

      // return output to pass as input to next node in graph
      return out_name;
    }

    const std::vector<Index> onnx_out_dims() override {
      return unfolded.vec_dims();
    }

  private:
    DeviceTensor<Scalar, 4, Device_, Layout> folded;
    DeviceTensor<Scalar, 2, Device_, Layout> unfolded;
    array<Index, 4> original_dimensions;

  };


}
