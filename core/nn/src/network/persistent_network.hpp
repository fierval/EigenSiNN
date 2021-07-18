#pragma once

#include "network_graph.hpp"
#include "onnx/loader.h"

namespace EigenSinn {

  template <typename Scalar, typename Actual, typename Loss, typename Device_>
  class PersistentNetworkBase : public NetworkBase<Scalar, Actual, Loss, Device_>
  {
  public:
    typedef std::shared_ptr<EigenModel> PtrEigenModel;

    PersistentNetworkBase() {

    }

    inline PtrEigenModel save(bool for_training = true) {

      // need to make sure we have topologically sorted vertices
      assert(forward_order.size() > 0);
      PtrEigenModel model = std::make_shared<EigenModel>(for_training);

      vertex_name names = boost::get(boost::vertex_name, graph);

      // we need to save the model based on the topological order
      // or else we won't be able to define edges in the ONNX graph
      for (auto& v : forward_order) {

        std::string v_name = names(v);
        std::map<std::string, std::string> layer_output;

        auto input_names = collect_input_names(v);

        // input layer.
        // the "output" of the input layer is the input layer itself
        if (input_names.empty()) {
          add_layer_output(layer_output, v_name, v_name);
          continue;
        }

        // collect output names: these are input edges into the layer
        // which are named after the input tensors
        std::vector<std::string> tensor_input_names(input_names.size());

        std::transform(input_names.begin(), input_names.end(), tensor_input_names.begin(), [&](std::string& in_layer) {
          return layer_output[in_layer];
          });

        // we'll get back the name of the output edge for this tensor and use it to
        // keep building the graph
        std::string layer_output_name = graph[v].layer->add_onnx_node(*model, tensor_input_names);

        add_layer_output(layer_output, v_name, layer_output_name);
      }

      return model;
    }

    inline void load(EigenModel& model, bool weights_only = false) {

      OnnxLoader<Scalar, Device_> onnx_layers(model);
      onnx::GraphProto& onnx_graph = onnx_layers.get_onnx_graph();

      // structures for loading onnx graph
      std::map<std::string, PtrLayer> name_layer_map;
      std::map<std::string, std::string> output_of_layer;
      std::map<std::string, StringVector> layer_s_inputs;

      for (auto& node : onnx_graph.node()) {
        StringVector inputs;
        std::string output;
        PtrLayer layer;

        std::tie(inputs, output, layer) = onnx_layers.create_from_node(node);
        layer->set_cudnn(true);

        std::string name = layer->get_layer_name();

        name_layer_map.insert(std::make_pair(name, layer));
        output_of_layer.insert(std::make_pair(output, name));
        layer_s_inputs.insert(std::make_pair(name, inputs));
      }
      
      if (!weights_only) {
        clear();
        add_vertices(name_layer_map, output_of_layer, layer_s_inputs);
        add_edges(name_layer_map, output_of_layer, layer_s_inputs);
        return;
      }

      // otherwise just add weights
      for (auto& p : name_layer_map) {
        std::string name = p.first;
        PtrLayer layer = p.second;

          vertex_t v = vertices[name];
          graph[v].layer.reset(layer.get());
      }
    }

  private:

    std::vector<std::string> collect_input_names(vertex_t& v) {

      std::vector<std::string> input_names;
      InEdgeIter in, in_end;
      vertex_name names = boost::get(boost::vertex_name, graph);

      for (boost::tie(in, in_end) = boost::in_edges(v, graph); in != in_end; in++) {

        vertex_t inp = boost::source(*in, graph);
        input_names.push_back(names[inp]);
      }

      return input_names;
    }

    // add name of the output to the map if it is not yet there
    void add_layer_output(std::map<std::string, std::string>& layer_output, std::string& layer_name, std::string& output_name) {
      if (layer_output.count(layer_name) > 0) {
        return;
      }
      layer_output.insert(std::make_pair(layer_name, output_name));
    }

    bool is_input_layer(const std::string& input_name) {
      return input_name.rfind(input_op, 0) == 0;
    }

    void add_vertices(std::map<std::string, PtrLayer>& name_layer_map, std::map<std::string, std::string>& output_of_layer, std::map<std::string, StringVector>& layer_s_inputs) {

      // add all the vertices don't forget the inputs
      // not reflected in the name_layer_map
      for (auto& p : name_layer_map) {
        std::string name = p.first;
        PtrLayer layer = p.second;

        auto v = boost::add_vertex(VertexProperty(name), graph);
        vertices.insert(std::make_pair(name, v));
        graph[v].layer = layer;

        // check if the layer has inputs and add them
        StringVector inputs = get_layer_inputs(layer_s_inputs, output_of_layer, name);
        StringVector input_layer_names;
        std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(input_layer_names), [this](std::string& s) {return this->is_input_layer(s); });
        for(auto& in_name : input_layer_names) {

          auto inp_layer = new Input<Scalar, Device_>;
          inp_layer->set_layer_name(in_name);
          add(inp_layer);
        }
      }
    }

    void add_edges(std::map<std::string, PtrLayer>& name_layer_map, std::map<std::string, std::string>& output_of_layer, std::map<std::string, StringVector>& layer_s_inputs) {

      // at this point all the vertices are in-place. Just need to connect them
      // all inputs have also been added so all we need is go through each vertex and connect them
      // name_layer_map does not contain input layers so no need to check
      for (auto& p : name_layer_map) {
        std::string name = p.first;

        auto inputs = get_layer_inputs(layer_s_inputs, output_of_layer, name);
        for (auto& i : inputs) {
          auto v_in = vertices[i];
          auto v_out = vertices[name];
          boost::add_edge(v_in, v_out, graph);
        }
      }
    }

    StringVector get_layer_inputs(std::map<std::string, StringVector>& layer_s_inputs, std::map<std::string, std::string>& output_of_layer, std::string& layer_name) {

      // find what layer outputs this input and add it first
      StringVector& inputs = layer_s_inputs[layer_name];
      StringVector true_inputs;

      // a true input is an output of some other layer
      std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(true_inputs), [&](std::string& s) {return output_of_layer.count(s) > 0; });
      return true_inputs;
    }
  };
}