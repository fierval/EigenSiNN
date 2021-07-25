#pragma once

#include <ops/opsbase.hpp>
#include <fstream>
#include <google/protobuf/text_format.h>
#include "onnx.proto3.pb.h"
#include <stdexcept>

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
    }

    EigenModel(const std::string& data, bool _is_training = true) 
      : is_training(_is_training)    {

      model.ParseFromString(data);

    }

    EigenModel(EigenModel& m) {

      model = m.model;
      current_value_name = m.current_value_name;
      current_layer_suffix = m.current_layer_suffix;
      is_training = m.is_training;
    }


    EigenModel& operator=(const EigenModel& m) {
      if (this == &m) { return *this; }

      model = m.model;
      current_value_name = m.current_value_name;
      current_layer_suffix = m.current_layer_suffix;
      is_training = m.is_training;

      return *this;
    }

    // add input/output tensor descriptors to the graph
    inline void add_input(const std::string& name, std::vector<Index>& dims, onnx::TensorProto_DataType data_type) {

      auto graph = get_graph();
      auto value_proto = graph->add_input();
      //auto* v = new onnx::ValueInfoProto();
      add_value_proto(name, value_proto, dims, data_type);
      //add_value_proto(name, v, dims, data_type);
    }

    // extract dimensions of the model input from model properties
    inline std::vector<Index> get_input_dims(const std::string& input_name) {

      auto graph = get_graph();
      std::vector<onnx::ValueInfoProto> value_protos(1);
       
      std::copy_if(graph->value_info().begin(), graph->value_info().end(), value_protos.begin(), [&](const onnx::ValueInfoProto& v) {
        return v.name() == input_name;
        });
      
      auto value_proto = value_protos[0];

      onnx::TypeProto type = value_proto.type();

      auto tensor_type = type.tensor_type();

      onnx::TensorShapeProto shape = tensor_type.shape();
      std::vector<Index> out_dims(shape.dim_size());
      std::transform(shape.dim().begin(), shape.dim().end(), out_dims.begin(), [](const onnx::TensorShapeProto_Dimension& d) {return d.dim_value(); });
      return out_dims;
    }

    inline void add_output(const std::string& name, const std::vector<Index>& dims, onnx::TensorProto_DataType data_type) {

      auto graph = get_graph();
      auto value_proto = graph->add_output();
      add_value_proto(name, value_proto, dims, data_type);
    }

    inline void add_output(const std::string& name, onnx::TensorProto_DataType data_type) {

      auto graph = get_graph();
      auto value_proto = graph->add_output();
      add_value_proto(name, value_proto, data_type);
    }

    inline void add_value_proto(const std::string& name, onnx::ValueInfoProto* value_proto, onnx::TensorProto_DataType data_type) {
      value_proto->set_name(name);
      onnx::TypeProto* type = value_proto->mutable_type();

      auto* tensor_type = type->mutable_tensor_type();
      tensor_type->set_elem_type(data_type);

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
    inline std::string get_tensor_value_name() { 

      return std::to_string(current_value_name++); 
    }

    // an idempotent (hopefully) way of getting output names from the inputs
    inline std::string get_tensor_value_name(std::vector<std::string>& input_names) {
      std::vector<int> input_ints;

      std::transform(input_names.begin(), input_names.end(), std::back_inserter(input_ints), [](std::string& name) {
        std::stringstream ss(name);
        int i = 0;
        ss >> i;
        return i;
        });

      int max_inp = *std::max_element(input_ints.begin(), input_ints.end());

      // REVIEW: what if we are branching? Should be ok as long we are traversing in topological order
      if (max_inp > 0) {
        current_value_name = max_inp + 2;
        return std::to_string(max_inp + 1);
      }
      return input_names[input_names.size() - 1] + "_1";
    }

    inline std::vector<std::string> get_cool_display_tensor_value_names(const char* prefix, std::vector<const char*> suffixes) {

      int layer_idx = get_layer_suffix();

      std::vector<std::string> out(suffixes.size());
      for (int i = 0; i < out.size(); i++) {
        out[i] = get_cool_display_tensor_value_name(prefix, layer_idx, suffixes[i]);
      }
      return out;
    }

    inline int get_layer_suffix() {
      return current_layer_suffix++;
    }

    // adds a graph node with descriptions and references to inputs/outputs
    inline onnx::NodeProto * add_graph_node(const std::string& layer_name, const std::string& op_type, const std::string& input_name) {

      std::vector<std::string> names{ input_name };
      return add_graph_node(layer_name, op_type, names);
    }

    inline onnx::NodeProto* add_graph_node(const std::string& layer_name, const std::string& op_type, std::vector<std::string>& input_names) {

      auto graph = get_graph();
      onnx::NodeProto * node = graph->add_node();

      node->set_name(layer_name);
      node->set_op_type(op_type);

      std::for_each(input_names.begin(), input_names.end(), [&](std::string& n) {
        auto * input = node->add_input();
        *input = n;
        });

      std::string * output = node->add_output();
      *output = get_tensor_value_name(input_names);

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

    static inline EigenModel FromFile(const std::string& file_name) {

      std::ifstream in(file_name, std::ofstream::binary);

      EigenModel m;
      m.model.ParseFromIstream(&in);

      return m;
    }

    static inline EigenModel FromString(const std::string& str) {
      EigenModel m;

      m.model.ParseFromString(str);
      return m;
    }

    inline std::string to_string() {
      std::string data;
      gp::TextFormat::PrintToString(model, &data);
      return data;
    }

    inline std::string flush_to_string() {

      return model.SerializeAsString();
    }

    inline onnx::TensorProto find_initializer(const std::string name) {

      auto graph = *get_graph();

      auto initializers = graph.initializer();
      auto it = initializers.begin();
      for (; it != initializers.end(); it++) {
        if (it->name() == name) {
          break;
        }
      }

      if (it == initializers.end()) {
        throw std::logic_error("Name not found in ONNX file");
      }
      return *it;
    }

    // given input name find its dimensions
    // REVIEW: NVCC doesn't like std::string&!!
    std::vector<Index> get_input_dimensions(std::string input_name) {

      const onnx::TensorProto  initializer = find_initializer(input_name);
      return get_input_dimensions(initializer);
    }

    std::vector<Index> get_input_dimensions(const onnx::TensorProto& initializer) {

      std::vector<Index> out(initializer.dims_size());
      std::transform(initializer.dims().begin(), initializer.dims().end(), out.begin(), [](int i) { return i; });
      return out;
    }

    template<typename Scalar>
    Scalar* get_input_data(onnx::TensorProto& initializer) {

      const char * data = initializer.raw_data().c_str();
      return (Scalar*)data;
    }

    template<typename Scalar>
    std::tuple<std::vector<onnx::TensorProto>, std::vector<std::vector<Index>>> get_input_data_and_dimensions(std::vector<std::string>& inputs) {

      // get all the inputs not counting the previous layer input
      std::vector<onnx::TensorProto> initializers = get_all_initializers(inputs);
      std::vector<std::vector<Index>> dimensions(initializers.size());

      std::transform(initializers.begin(), initializers.end(), dimensions.begin(),
        [&](onnx::TensorProto& i) {return get_input_dimensions(i); });

      return std::make_tuple(initializers, dimensions);
    }

    std::vector<onnx::TensorProto> get_all_initializers(std::vector<std::string>& inputs) {

      std::vector<onnx::TensorProto> initializers(inputs.size() - 1);

      std::transform(inputs.begin() + 1, inputs.end(), initializers.begin(),
        [&](std::string& name) {return find_initializer(name); });

      return initializers;
    }

    template<typename T>
    void add_attr(onnx::NodeProto * node, const std::string& name, T value) {

      if (name.empty()) { throw std::runtime_error("Empty attribute name"); }

      auto* new_attr = node->add_attribute();
      new_attr->set_name(name);

      if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
        new_attr->set_type(onnx::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT);
        new_attr->set_f(value);
        return;
      }

      if (std::is_same<T, int>::value || std::is_same<T, Index>::value) {
        new_attr->set_type(onnx::AttributeProto::AttributeType::AttributeProto_AttributeType_INT);
        new_attr->set_i(value);
        return;
      }
      throw std::runtime_error("Unkonwn attribute type");
    }

  private:

    // an alternative to the above when we want input names
    // to be more descriptive
    inline std::string get_cool_display_tensor_value_name(const char* prefix, int layer_idx, const char* suffix) {

      std::ostringstream ss;
      ss << prefix << layer_idx << "." << suffix;
      return ss.str();

    }

    // current name for tensor value
    int current_value_name = 400;
    int current_layer_suffix = 0;

    onnx::ModelProto model;
    bool is_training;
  };

} // namespace EigenSinn