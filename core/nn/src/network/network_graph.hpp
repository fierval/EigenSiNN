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

    PtrLayer<Scalar, Device_> layer;

    // to be able to store it in a set
    inline bool operator< (const VertexData d) {
      return layer->get_layer_name() < d.layer->get_layer_name();
    }

    VertexData(LayerBase<Scalar, Device_>* _layer)
      : layer(_layer) {

    }
  };

  template <typename Scalar, typename Device_>
  using NetworkGraph = boost::adjacency_list<boost::setS, boost::vecS, boost::directedS,
    boost::property<boost::vertex_name_t, std::string, VertexData<Scalar, Device_>>;

  template <typename Scalar, typename Device_>
  using OptionalNetworkGraph = std::optional<NetworkGraph<Scalar, Device_>>;

  template <typename Scalar, typename Device_>
  class NetworkBase {

    typedef std::vector<VertexData<Scalar, Device_>> VertexList;

  public:
    NetworkBase() {
    }

    // seed the graph with network inputs
    std::string add(Input<Scalar, Device_>* inp_layer) {

      // we've been here before!
      if (!inp_layer->get_layer_name().empty()) {
        return inp_layer->get_layer_name();
      }

      inp_layer->set_layer_name(get_layer_name());
      std::string out_name(get_out_name());
      input_vertices.push_back(VertexData<Scalar, Device_>(inp_layer));
      graph_capacity++;

    }

    // add one edge to the future graph
    std::string add(const std::string& input_name, LayerBase<Scalar, Device_>* layer) {

      // name the current vertex if it doesn't yet have a name
      if (layer->get_layer_name().empty()) {

        layer->set_layer_name(get_layer_name());
        net_vertices.insert(layer->get_layer_name(), VertexData<Scalar, Device_>(layer));
      }
      
      // crate the vertex & insert it into the map
      VertexData<Scalar, Device_> vertex = net_vertices[layer->get_layer_name()];

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
      for (auto& inp_to_vertex : vertices)
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

    std::string get_layer_name() {
      std::stringstream opname_stream;
      opname_stream << layer->get_op_name() << "_" << get_current_layer_suffix();
      return opname_stream.str();
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
    std::map<std::string, VertexList> vertices;

    // this is how we "seed" the edges
    std::vector<VertexList> input_vertices;
    std::map<std::string, VertexData<Scalar, Device_>> net_vertices;

    int graph_capacity = 0;
  };

} // namespace EigenSinn