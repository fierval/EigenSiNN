#pragma once

#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/named_function_params.hpp>

#include <layers/layer_base.hpp>
#include <layers/input.hpp>
#include <optimizers/optimizer_base.hpp>

// the predefined property types are listed here:
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/using_adjacency_list.html#sec:adjacency-list-properties
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/bundles.html

namespace EigenSinn {
  
  template <typename Scalar, typename Device_>
  using PtrLayer = std::shared_ptr<LayerBase<Scalar, Device_>>;

  template <typename Scalar, typename Device_>
  using PtrOptimizer = std::shared_ptr<OptimizerBase<Scalar, Device_>>;

  // vertex properties
  template <typename Scalar, typename Device_>
  struct VertexData {
    std::string op_name; //layer or operation
    PtrLayer<Scalar, Device_> layer;

    std::string out_name;

    VertexData(const std::string& _op_name, LayerBase<Scalar, Device_> * _layer, std::string& _out_name)
      : op_name(_op_name)
      , layer(_layer)
      , out_name(_out_name)  {
        
    }
  };

  template <typename Scalar, typename Device_>
  using NetworkGraph =  boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexData<Scalar, Device_>>;

  template <typename Scalar, typename Device_>
  using OptionalNetworkGraph = std::optional<NetworkGraph<Scalar, Device_>>;

  template <typename Scalar, typename Device_>
  class NetworkBase {

    typedef std::vector<VertexData<Scalar, Device_>> VertexList;

  public:
    NetworkBase() {
    }

    std::string add(Input<Scalar, Device_>* inp_layer) {
      return std::string();
    }

    // add one edge to the future graph
    std::string add(const std::string& input_name, LayerBase<Scalar, Device_>* layer) {

      // name the current vertex if it doesn't yet have a name
      if (get_layer_name().empty()) {
        std::stringstream opname_stream;
        opname_stream << layer->get_op_name() << "_" << get_current_layer_suffix();

        layer->set_layer_name(opname_stream.str());
      }

      // name output tensor
      std::string out_name(get_out_name());

      // crate the vertex & insert it into the map
      VertexData<Scalar, Device_> vertex(layer->get_layer_name(), layer, out_name);

      if (vertices.find(input_name) == vertices.end()) {
        vertices.insert(vertex, std::vector< VertexData<Scalar, Device_>>);
      }

      graph_capacity++;
      vertices[input_name].push_back(vertex);

      return out_name;
    }

    // walk the vertices and create an actual graph
    void compile() {

      if (graph.has_value()) {
        throw std::logic_error("Graph already created");
      }

      create_graph();
      for(auto& inp_to_vertex : vertices)
      {
        VertexList& out_vertices = inp_to_vertex.second;
        
      }
    }

  protected:
    int get_current_layer_suffix() { return current_name_suffix++; }
    std::string get_out_name() {

      std::stringstream ss;
      ss << "out_" << current_output_suffix++;
      return ss.str();
    }

    void add_vertices(VertexData<Scalar, Device_>& vertex, VertexList& out_vertices) {

      assert(graph);
      std::for_each(out_vertices.begin(), out_vertices.end(), [&](VertexData<Scalar, Device_>& v) {
        boost::add_edge(vertex, v, *grpah);
        });
    }

    void create_graph() {
      assert(!graph.has_value());
      assert(vertices.size() > 0);
      assert(graph_capacity > 0);

      graph = OptionalNetworkGraph<Scalar, Device_>(graph_capacity);
    }

  private:
    OptionalNetworkGraph<Scalar, Device_> graph;
    int current_name_suffix = 1;
    int current_output_suffix = 400;

    // structure to build the graph from
    std::map<std::string, std::vector<VertexData<Scalar, Device_>>> vertices;
    
    int graph_capacity = 0;
  };

} // namespace EigenSinn