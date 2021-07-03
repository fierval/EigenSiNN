#pragma once

#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/named_function_params.hpp>

#include <layers/layer_base.hpp>
#include <optimizers/optimizer_base.hpp>

// the predefined property types are listed here:
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/using_adjacency_list.html#sec:adjacency-list-properties
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/bundles.html

namespace EigenSinn {
  
  template <typename Scalar, typename Device_>
  using PtrLayer = std::shared_ptr<LayerBase<Scalar, Device_>>;

  template <typename Scalar, typename Device_>
  using PtrOptimizer = std::shared_ptr<OptimizerBase<Scalar, Device_>>;

  typedef std::pair<std::string, std::string> InOut;

  template <typename Scalar, typename Device_>
  struct VertexData {
    std::string op_name; //layer or operation
    PtrLayer<Scalar, Device_> layer;

    InOut input_out_names;

    VertexData(const std::string& _op_name, LayerBase<Scalar, Device_> * _layer, const InOut& _in_out_name)
      : op_name(_op_name)
      , layer(_layer)
      , input_out_names(_in_out_name)  {
        
    }
  };

  template <typename Scalar, typename Device_>
  using NetworkGraph =  boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexData<Scalar, Device_>>;

  template <typename Scalar, typename Device_>
  using OptionalNetworkGraph = std::optional<NetworkGraph<Scalar, Device_>>;

  template <typename Scalar, typename Device_>
  class NetworkBase {

  public:
    NetworkBase() {
    }

    // add one edge to the future graph
    template<typename OptimizerType>
    std::string add(const std::string& input_name, LayerBase<Scalar, Device_>* layer) {

      // name the current vertex if it doesn't yet have a name
      if (get_layer_name().empty()) {
        std::stringstream opname_stream;
        opname_stream << layer->get_op_name() << "_" << get_current_layer_suffix();

        layer->set_layer_name(opname_stream.str());
      }

      // name output tensor
      std::string out_name(get_out_name());
      
      // create an edge
      InOut edge = std::make_pair(input_name, out_name);
      in_out_map.insert(edge);

      VertexData<Scalar, Device_> vertex(layer->get_layer_name(), layer, edge);
      graph_size++;


      return std::string();
    }

  protected:
    int get_current_layer_suffix() { return current_name_suffix++; }
    std::string get_out_name() {

      std::stringstream ss;
      ss << "out_" << current_output_suffix++;
      return ss.str();
    }

    void create_graph() {
      assert(!graph.has_value());
      assert(graph_size > 0);

      graph = OptionalNetworkGraph<Scalar, Device_>(graph_size);
    }

  private:
    OptionalNetworkGraph<Scalar, Device_> graph;
    int current_name_suffix = 1;
    int current_output_suffix = 400;

    int graph_size = 0;

    // structure to build the graph from
    // TODO: multiple outputs?
    std::map <std::string, VertexData<Scalar, Device_>> vertex_map;
   
  };

} // namespace EigenSinn