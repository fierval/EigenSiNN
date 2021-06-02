#pragma once

#include <ops/opsbase.hpp>
#include <fstream>
#include <google/protobuf/text_format.h>
#include "onnx.proto3.pb.h"

namespace gp = google::protobuf;

using namespace Eigen;

namespace EigenSinn {


  template<typename Scalar>
  onnx::TensorProto_DataType onnx_data_type_from_scalar() {

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
      
      onnx::OperatorSetIdProto * opset_id = model.add_opset_import();

      // https://github.com/onnx/onnx/blob/master/docs/Versioning.md
      opset_id->set_version(14);

      model.set_ir_version(onnx::Version::IR_VERSION);

      std::string docstring = "EigenSiNN Model Format";
      model.set_doc_string(docstring);

      std::string producer = "EigenSiNN";
      std::string version = "0.1.2";

      model.set_producer_name(producer);
      model.set_producer_version(version);

      // allocate graph & pass ownership to model
      auto graph = model.mutable_graph();
    }

    // add input/output tensor descriptors to the graph
    inline void add_input(const std::string& name, std::vector<Index>& dims, onnx::TensorProto_DataType data_type) {

      auto graph = get_graph();
      auto value_proto = graph->add_input();
      //auto* v = new onnx::ValueInfoProto();
      add_value_proto(name, value_proto, dims, data_type);
      //add_value_proto(name, v, dims, data_type);
    }

    inline void add_output(const std::string& name, const std::vector<Index>& dims, onnx::TensorProto_DataType data_type) {

      auto graph = get_graph();
      auto value_proto = graph->add_output();
      add_value_proto(name, value_proto, dims, data_type);
    }

    inline void add_value_proto(const std::string& name, onnx::ValueInfoProto* value_proto, const std::vector<Index>& dims, onnx::TensorProto_DataType data_type) {

      value_proto->set_name(name);
      onnx::TypeProto* type = value_proto->mutable_type();

      auto * tensor_type = type->mutable_tensor_type();
      tensor_type->set_elem_type(data_type);

      onnx::TensorShapeProto * shape = tensor_type->mutable_shape();
      for (int i = 0; i < dims.size(); i++) {
        auto * dim = shape->add_dim();
        dim->set_dim_value(dims[i]);
      }
    }

    // get a unique name for the tensor value
    static inline std::string get_tensor_value_name() { 

      std::lock_guard<std::mutex> guard(value_mutex);
      return std::to_string(current_value_name++); 
    }

    static inline std::vector<std::string> get_cool_display_tensor_value_names(const char* prefix, std::vector<const char*> suffixes) {

      int layer_idx = EigenModel::get_layer_suffix();

      std::vector<std::string> out(suffixes.size());
      for (int i = 0; i < out.size(); i++) {
        out[i] = EigenModel::get_cool_display_tensor_value_name(prefix, layer_idx, suffixes[i]);
      }
      return out;
    }

    static inline int get_layer_suffix() {
      std::lock_guard<std::mutex> guard(suffix_mutex);
      return current_layer_suffix++;
    }

    // adds a graph node with descriptions and references to inputs/outputs
    inline onnx::NodeProto * add_graph_node(const char* op_type, const std::string& input_name) {

      std::vector<std::string> names{ input_name };
      return add_graph_node(op_type, names);
    }

    inline onnx::NodeProto* add_graph_node(const char* op_type, std::vector<std::string>& input_names) {

      std::ostringstream layer_name_stream;
      layer_name_stream << op_type << "_" << get_layer_suffix();

      auto graph = get_graph();
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

    inline onnx::GraphProto * get_graph() { return model.mutable_graph(); }

    // some layers like Dropout aren't needed if we are saving
    // for inference
    inline bool is_inference() { return !is_training; }

    // serialize to file
    inline void flush(const std::string& file_name) {
      std::ofstream out(file_name, std::ofstream::binary);

      model.SerializeToOstream(&out);
    }

    // dump text of the model to a text file (good for debugging)
    inline void dump(const std::string& file_name) {

      std::ofstream out(file_name);

      std::string data;
      gp::TextFormat::PrintToString(model, &data);
      out << data;
    }

    inline std::string to_string() {
      std::string data;
      gp::TextFormat::PrintToString(model, &data);
      return data;
    }


    static inline std::shared_ptr<EigenModel> LoadOnnxModel(const std::string& data) {

      onnx::ModelProto _model;
      instance = std::make_shared<EigenModel>(std::move(_model));
      instance->model.ParseFromString(data);

      return instance;
    }

  private:

    // ctor for parsing
    EigenModel(onnx::ModelProto&& _model) : model(_model) {
      
    }

    // instance for parsing
    static inline std::shared_ptr<EigenModel> instance;

    // an alternative to the above when we want input names
    // to be more descriptive
    static inline std::string get_cool_display_tensor_value_name(const char* prefix, int layer_idx, const char* suffix) {

      std::ostringstream ss;
      ss << prefix << layer_idx << "." << suffix;
      return ss.str();

    }

    // current name for tensor value
    static inline int current_value_name = 400;
    static inline int current_layer_suffix = 0;
    static inline std::mutex value_mutex;
    static inline std::mutex suffix_mutex;

    onnx::ModelProto model;
    bool is_training;
  };

} // namespace EigenSinn