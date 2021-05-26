#pragma once

#include <ops/opsbase.hpp>
#include <layers/input.hpp>
#include "onnx.proto3.pb.h"

namespace EigenSinn {

  template<int Rank>
  std::vector<Index> dsizes2vector(DSizes<Index, Rank> dims) {

    std::vector<Index> out(Rank);
    for (int i = 0; i < Rank; i++) {
      out[i] = dims[i];
    }
    
    return out;
  }

  template<typename Scalar>
  onnx::TensorProto_DataType data_type_from_scalar() {

    if (std::is_same<Scalar, float>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT;
    }

    if (std::is_same<Scalar, double>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE;
    }

    if (std::is_same<Scalar, int>::value || std::is_same<Scalar, long>::value || std::is_same<Scalar, __int32>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_INT32;
    }

    if (std::is_same<Scalar, long long>::value || std::is_same<Scalar, __int64>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_INT64;
    }

    if (std::is_same<Scalar, short>::value || std::is_same<Scalar, __int16>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_INT16;
    }

    if (std::is_same<Scalar, char>::value || std::is_same<Scalar, __int8>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_INT8;
    }

    if (std::is_same<Scalar, unsigned __int32>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT32;
    }

    if (std::is_same<Scalar, unsigned __int16>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT16;
    }

    if (std::is_same<Scalar, unsigned char>::value) {
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT8;
    }

    assert(false);
  }

  class EigenModel {

    typedef std::vector<onnx::ValueInfoProto *> ValueInfos;

  public:
    EigenModel() {
      
      model = std::make_shared<onnx::ModelProto>();
      onnx::OperatorSetIdProto * opset_id = model->add_opset_import();

      // https://github.com/onnx/onnx/blob/master/docs/Versioning.md
      opset_id->set_version(14);

      model->set_ir_version(onnx::Version::IR_VERSION);

      std::string docstring = "EigenSiNN Model Format";
      model->set_doc_string(docstring);

      std::string producer = "EigenSiNN";
      std::string version = "0.1.2";

      model->set_producer_name(producer);
      model->set_producer_version(version);

      // allocate graph & pass ownership to model
      graph = new onnx::GraphProto();
      model->set_allocated_graph(graph);

    }

    // add input/output tensor descriptors to the graph
    inline void add_input(const std::string& name, std::vector<Index>& dims, onnx::TensorProto_DataType data_type) {

      auto value_proto = graph->add_input();
      add_value_proto(name, value_proto, dims, data_type);
    }

    inline void add_output(const std::string& name, std::vector<Index>& dims, onnx::TensorProto_DataType data_type) {

      auto value_proto = graph->add_output();
      add_value_proto(name, value_proto, dims, data_type);
    }

    inline void add_value_proto(const std::string& name, onnx::ValueInfoProto* value_proto, std::vector<Index>& dims, onnx::TensorProto_DataType data_type) {

      value_proto->set_name(name);
      onnx::TypeProto* type = new onnx::TypeProto();
      value_proto->set_allocated_type(type);

      onnx::TypeProto_Tensor *tensor_type = new onnx::TypeProto_Tensor();
      tensor_type->set_elem_type(data_type);
      onnx::TensorShapeProto * shape  = tensor_type->mutable_shape();

      for (int i = 0; i < dims.size(); i++) {
        shape->add_dim()->set_dim_value(dims[i]);
      }
    }

  protected:
    std::shared_ptr<onnx::ModelProto> model;
    onnx::GraphProto * graph;
  };

} // namespace EigenSinn