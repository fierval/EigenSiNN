#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/config.hpp>
#include <boost/graph/graph_utility.hpp>

#include <layers/all.hpp>

// the predefined property types are listed here:
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/using_adjacency_list.html#sec:adjacency-list-properties
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/bundles.html

namespace EigenSinn {

  template <typename Scalar, typename Device_>
  using PtrLayer = std::shared_ptr<LayerBase<Scalar, Device_>>;

  // vertex properties
  template <typename Scalar, typename Device_>
  struct VertexData {

    PtrLayer<Scalar, Device_> layer;
  };

  template <typename Scalar, typename Device_>
  class NetworkBase {

    typedef boost::property <boost::vertex_name_t, std::string, VertexData<Scalar, Device_>> VertexProperty;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperty>
      NetworkGraph;

    // property map
    typedef typename boost::property_map<NetworkGraph, boost::vertex_name_t >::type vertex_name;

    // vertex descriptor
    typedef typename boost::graph_traits<NetworkGraph>::vertex_descriptor vertex_t;

  public:
    NetworkBase() {
    }

    // seed the graph with network inputs
    // walk the vertices and create an actual graph
    void compile() {

    }

    void print_graph() {
      vertex_name name = get(boost::vertex_name, graph);
      boost::print_graph(graph, name);
    }

  protected:
    int get_current_layer_suffix() { return current_name_suffix++; }

    std::string get_layer_name(const std::string& op_name) {
      std::stringstream opname_stream;
      opname_stream << op_name << "_" << get_current_layer_suffix();
      return opname_stream.str();
    }

    std::string& name_the_layer(LayerBase<Scalar, Device_>* layer) {
      layer->set_layer_name(get_layer_name(layer->get_op_name()));
      return layer->get_layer_name();
    }

    std::string add(Input<Scalar, Device_>* inp_layer) {

      std::string& inp_layer_name = inp_layer->get_layer_name();

      // we've been here before!
      if (!inp_layer_name.empty()) {
        assert(vertices.count(inp_layer_name) > 0);
        return inp_layer_name;
      }

      inp_layer_name = name_the_layer(inp_layer);

      vertex_t v = boost::add_vertex(VertexProperty(inp_layer_name), graph);
      graph[v].layer.reset(inp_layer);

      vertices.insert(std::make_pair(inp_layer_name, v));
      return inp_layer_name;
    }

    // add one edge to the future graph
    std::string add(const std::string& input_name, LayerBase<Scalar, Device_>* layer) {

      assert(vertices.count(input_name) > 0);
      std::string& layer_name = layer->get_layer_name();
      vertex_t v_out, v_in;

      // name the current vertex if it doesn't yet have a name
      if (layer_name.empty()) {
        layer_name = name_the_layer(layer);

        assert(vertices.count(layer_name) == 0);

        v_out = boost::add_vertex(VertexProperty(layer_name), graph);
        graph[v_out].layer.reset(layer);

        vertices.insert(std::make_pair(layer_name, v_out));
      }

      v_out = vertices[layer_name];
      v_in = vertices[input_name];
      boost::add_edge(v_in, v_out, graph);

      std::cerr << "Layer property: " << graph[v_in].layer->get_layer_name() << " to " << graph[v_out].layer->get_layer_name() << std::endl;
      return layer_name;
    }

  private:
    NetworkGraph graph;
    int current_name_suffix = 1;

    // structure to build the graph from
    std::map<std::string, vertex_t> vertices;
  };

} // namespace EigenSinn