#pragma once

#include <ops/opsbase.hpp>
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
    EigenModel(bool _is_training = true)
    : is_training(_is_training ) {
      
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

    // get a unique name for the tensor value
    static inline std::string get_tensor_value_name() { 

      std::lock_guard<std::mutex> guard(value_mutex);
      return std::to_string(current_value_name++); 
    }

    static inline std::string get_layer_suffix() {
      std::lock_guard<std::mutex> guard(suffix_mutex);
      return std::to_string(current_layer_suffix++);
    }

    // adds a graph node with descriptions and references to inputs/outputs
    inline onnx::NodeProto * add_graph_node(const char* op_type, const std::string& input_name) {

      std::vector<std::string> names{ input_name };
      return add_graph_node(op_type, names);
    }

    inline onnx::NodeProto* add_graph_node(const char* op_type, std::vector<std::string>& input_names) {

      std::ostringstream layer_name_stream;
      layer_name_stream << op_type << "_" << get_layer_suffix();
      
      onnx::NodeProto * node = graph->add_node();

      node->set_name(layer_name_stream.str());
      node->set_op_type(op_type);

      std::for_each(input_names.begin(), input_names.end(), [&](std::string& n) {
        auto * input = node->add_input();
        *input = n;
        });

      std::string * output = node->add_output();
      *output = get_tensor_value_name();

      return node;
    }

    inline onnx::GraphProto* get_graph() { return graph; }

    // some layers like Dropout aren't needed if we are saving
    // for inference
    inline bool is_inference() { return !is_training; }

  protected:
    
    // current name for tensor value
    static inline int current_value_name = 400;
    static inline int current_layer_suffix = 0;
    static inline std::mutex value_mutex;
    static inline std::mutex suffix_mutex;

    std::shared_ptr<onnx::ModelProto> model;
    onnx::GraphProto * graph;

    bool is_training;
  };

} // namespace EigenSinn