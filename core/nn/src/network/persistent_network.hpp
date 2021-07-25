#pragma once

#include "network_graph.hpp"
#include "onnx/loader.h"

namespace EigenSinn {

  template <typename Scalar, typename Actual, typename Loss, typename Device_>
  class PersistentNetworkBase : public NetworkBase<Scalar, Actual, Loss, Device_>
  {
  public:
    typedef std::shared_ptr<EigenModel> PtrEigenModel;
    typedef std::unordered_map<std::string, LayerBase<Scalar, Device_>*> LayerMap;
    typedef std::unordered_map<std::string, std::string> StringMap;
    typedef std::unordered_map<std::string, StringVector> StringVectorMap;

    PersistentNetworkBase() {

    }

    inline PtrEigenModel save(bool for_training = true) {

      // need to make sure we have topologically sorted vertices
      assert(forward_order.size() > 0);
      PtrEigenModel model = std::make_shared<EigenModel>(for_training);

      // keep track of outputs that become inputs to other layers.
      // if they don't - we have reached an output layer
      std::unordered_set<std::string> in_names;

      vertex_name names = boost::get(boost::vertex_name, graph);
      StringMap layer_output;

      // we need to save the model based on the topological order
      // or else we won't be able to define edges in the ONNX graph
      for (auto& v : forward_order) {

        std::string v_name = names(v);

        auto input_names = collect_input_names(v);

        // input layer.
        // the "output" of the input layer is the input layer itself
        if (input_names.empty()) {
          add_layer_output(layer_output, v_name, v_name);
          graph[v].layer->add_onnx_node(*model, v_name);
          continue;
        }

        // collect output names: these are input edges into the layer
        // which are named after the input tensors
        std::vector<std::string> tensor_input_names(input_names.size());

        std::transform(input_names.begin(), input_names.end(), tensor_input_names.begin(), [&](std::string& in_layer) {
          std::string& in_name = layer_output[in_layer];
          in_names.insert(in_name);
          return in_name;
          });

        // we'll get back the name of the output edge for this tensor and use it to
        // keep building the graph
        std::string layer_output_name = graph[v].layer->add_onnx_node(*model, tensor_input_names);

        add_layer_output(layer_output, v_name, layer_output_name);
      }

      // figure out which layers are actually "output" layers: not leading to any other layers
      for (auto& p : layer_output) {

        // this is leading somewhere, so not the network output
        if (in_names.find(p.second) != in_names.end()) {
          continue;
        }

        // found an output!
        auto v = vertices[p.first];
        auto layer = graph[v].layer;
        try {
          model->add_output(p.second, layer->onnx_out_dims(), onnx_data_type_from_scalar<Scalar>());
        }
        catch (...) {
          // TODO: we won't be able to output dimensions if this is saving right after loading
          // since we haven't stepped through the graph and outputs are empty
          model->add_output(p.second, onnx_data_type_from_scalar<Scalar>());
        }
      }


      return model;
    }

    inline void load(EigenModel& model) {

      OnnxLoader<Scalar, Device_> onnx_layers(model);
      onnx::GraphProto& onnx_graph = onnx_layers.get_onnx_graph();

      // structures for loading onnx graph
      LayerMap name_layer_map;
      StringMap output_of_layer;
      StringVectorMap layer_s_inputs;

      for (auto& node : onnx_graph.node()) {
        StringVector inputs;
        std::string output;
        LayerBase<Scalar, Device_>* layer;

        std::tie(inputs, output, layer) = onnx_layers.create_from_node(node);
        layer->set_cudnn(true);

        std::string name = layer->get_layer_name();

        name_layer_map.insert(std::make_pair(name, layer));
        output_of_layer.insert(std::make_pair(output, name));
        layer_s_inputs.insert(std::make_pair(name, inputs));

        // check if the layer has inputs reprsented by Input_ layers (from the data, not from other layers) and add them
        StringVector data_inputs = get_layer_inputs(layer_s_inputs, output_of_layer, name, true);
        for (auto& in_name : data_inputs) {
          auto inp_layer = new Input<Scalar, Device_>;
          inp_layer->set_layer_name(in_name);

          name_layer_map.insert(std::make_pair(in_name, inp_layer));
          // output of the input layer is the layer itself
          output_of_layer.insert(std::make_pair(in_name, in_name));
        }

      }

      // rebuild the model graph
      clear();
      add_vertices(name_layer_map, output_of_layer, layer_s_inputs);
      add_edges(name_layer_map, output_of_layer, layer_s_inputs);

      attach_losses(output_of_layer, layer_s_inputs);
      if (optimizer_type != Optimizers::None) {
        add_optimizers();
      }
      compile();
      return;
    }

  private:

    inline std::vector<std::string> collect_input_names(vertex_t& v) {

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
    inline void add_layer_output(StringMap& layer_output, std::string& layer_name, std::string& output_name) {
      if (layer_output.count(layer_name) > 0) {
        return;
      }
      layer_output.insert(std::make_pair(layer_name, output_name));
    }

    inline bool is_input_layer(const std::string& input_name) {
      return input_name.rfind(input_op, 0) == 0;
    }

    inline void add_vertices(LayerMap& name_layer_map, StringMap& output_of_layer, StringVectorMap& layer_s_inputs) {

      // add all the vertices don't forget the inputs
      // not reflected in the name_layer_map
      for (auto& p : name_layer_map) {
        std::string name = p.first;
        LayerBase<Scalar, Device_>* layer = p.second;

        auto v = boost::add_vertex(VertexProperty(name), graph);
        vertices.insert(std::make_pair(name, v));
        graph[v].layer.reset(layer);
        if (is_input_layer(name)) {
          input_vertices.push_back(name);
        }
      }
    }

    inline void add_edges(LayerMap& name_layer_map, StringMap& output_of_layer, StringVectorMap& layer_s_inputs) {

      // at this point all the vertices are in-place. Just need to connect them
      // all inputs have also been added so all we need is go through each vertex and connect them
      // name_layer_map does not contain input layers so no need to check
      for (auto& p : name_layer_map) {
        std::string name = p.first;

        auto inputs = get_layer_inputs(layer_s_inputs, output_of_layer, name);

        for (auto& i : inputs) {
          auto v_in = vertices[output_of_layer[i]];
          auto v_out = vertices[name];
          boost::add_edge(v_in, v_out, graph);
        }
      }
    }

    // figure out where the logits are and
    // attach losses
    void attach_losses(StringMap& output_of_layer, StringVectorMap& layer_s_inputs) {

      // an output which is not present in the inputs to another layer means this layer is a "logit"
      std::unordered_set<std::string> inputs_to_layer;
      for (auto& p : layer_s_inputs) {
        inputs_to_layer.insert(p.second.begin(), p.second.end());
      }


      for (auto& p : output_of_layer) {

        std::string output_name = p.first;
        std::string layer_name = p.second;

        if (inputs_to_layer.find(output_name) == inputs_to_layer.end()) {
          logits.push_back(layer_name);
        }
      }

      add_loss();
    }

    // actual_input_layers - do we want inputs coming from any and all layers(false), or from Input layers only (true)?
    inline StringVector get_layer_inputs(StringVectorMap& layer_s_inputs, StringMap& output_of_layer, std::string& layer_name, bool actual_input_layers = false) {

      // find what layer outputs this input and add it first
      StringVector& inputs = layer_s_inputs[layer_name];
      StringVector true_inputs;

      // a true input is an output of some other layer
      if (actual_input_layers) {
        std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(true_inputs), [&](std::string& s) {
          return this->is_input_layer(s);
          });
      }
      else {
        std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(true_inputs), [&](std::string& s) {
          return output_of_layer.count(s) > 0;
          });
      }
      return true_inputs;
    }
  };
}