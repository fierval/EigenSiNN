#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/topological_sort.hpp>

#include <layers/all.hpp>
#include <optimizers/all.hpp>
#include <losses/all.hpp>

// the predefined property types are listed here:
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/using_adjacency_list.html#sec:adjacency-list-properties
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/bundles.html

namespace EigenSinn {

  template <typename Scalar, typename Device_, typename Loss>
  class NetworkBase {

    typedef std::shared_ptr<LayerBase<Scalar, Device_>> PtrLayer;

    struct VertexData {

      PtrLayer layer;
    };

    typedef boost::property <boost::vertex_name_t, std::string, VertexData> VertexProperty;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, VertexProperty>
      NetworkGraph;

    // property map
    typedef typename boost::property_map<NetworkGraph, boost::vertex_name_t >::type vertex_name;

    // vertex descriptor
    typedef typename boost::graph_traits<NetworkGraph>::vertex_descriptor vertex_t;

    // edge iterators
    typedef typename boost::graph_traits<NetworkGraph>::out_edge_iterator OutEdgeIter;
    typedef typename boost::graph_traits<NetworkGraph>::in_edge_iterator InEdgeIter;

    // smart pointers to data
    typedef std::shared_ptr<Loss> PtrLoss;
    typedef std::shared_ptr<OptimizerBase<Scalar, Device_, RowMajor>> PtrOptimizer;
    typedef PtrTensorAdapter<Scalar, Device_> PtrTensor;
    typedef std::vector<PtrTensorAdapter<Scalar, Device_>> TensorVector;
        
  public:
    NetworkBase() {

    }

    void print_graph() {

      vertex_name name = get(boost::vertex_name, graph);
      boost::print_graph(graph, name);
    }

    void write_graphviz(std::ofstream& stream) {

      boost::dynamic_properties dp;
      dp.property("name", boost::get(boost::vertex_name, graph));

      boost::write_graphviz_dp(stream, graph, dp, std::string("name"));
    }

    void print_traversal(bool forward = true) {

      vertex_name name = get(boost::vertex_name, graph);

      if (forward) {
        std::for_each(forward_order.begin(), forward_order.end(), [&](vertex_t& v) {
          std::cerr << name[v] << std::endl;
          });
      }
      else {
        std::for_each(forward_order.rbegin(), forward_order.rend(), [&](vertex_t& v) {
          std::cerr << name[v] << std::endl;
          });
      }
    }

    void init() {
      assert(!inited);
      for (auto& v : forward_order) {
        graph[v].layer->init();
      }
      inited = true;
    }

    void forward() {
      // we should have topologically sorted the graph
      assert(vertices.size() > 0);

      if (!inited) {
        init();
      }

      for (auto& v : forward_order) {

        TensorVector inputs;
        InEdgeIter in, in_end;

        // input layer
        if (in == in_end) { continue; }

        for (boost::tie(in, in_end) = boost::in_edges(v, graph); in != in_end; in++) {

          vertex_t inp = boost::source(*in, graph);
          inputs.push_back(graph[inp].layer->get_output());
        }

        // TODO: enable multi-input layers!!!
        PtrLayer current_layer = graph[v].layer;
        current_layer->forward(inputs);
      }

    }

    std::vector<std::string>& get_input_names() {
      return input_vertices;
    }

  protected:

    // single input network
    void set_input(PtrTensor& input) {
      
      assert(input_vertices.size() == 1);
      graph[vertices[input_vertices[0]]].layer->set_input(input);
    }

    void set_input(std::map<std::string, PtrTensor>& inputs) {

      assert(inputs.size() > 0);
      std::for_each(inputs.begin(), inputs.end(), [&](std::pair<std::string, PtrTensor> p) {
        graph[vertices[p.first]].layer->set_input(p.second);
        });
    }

    // walk the vertices and attach the optimizer
    // add loss function
    void add_loss_and_complete(const std::string& logits) {
      this->name_loss = std::make_pair(logits, std::make_shared<Loss>());

      // we will reverse-iterate during backprop
      boost::topological_sort(graph, std::front_inserter(forward_order));
    }

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

      auto name_vertex = std::make_pair(inp_layer_name, v);
      vertices.insert(name_vertex);
      input_vertices.push_back(inp_layer_name);

      return inp_layer_name;
    }

    // add one edge to the future graph
    // for now we need to instantiate the optimizer together with the vertex.
    std::string add(const std::string& input_name, LayerBase<Scalar, Device_>* layer, OptimizerBase<Scalar, Device_, RowMajor>* optimizer = nullptr) {

      assert(vertices.count(input_name) > 0);

      assert((layer->is_optimizable() && optimizer != nullptr) || (!layer->is_optimizable() && optimizer == nullptr));

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

      if (optimizer != nullptr) {
        optimizers.insert(std::make_pair(layer_name, PtrOptimizer(optimizer)));
      }

      return layer_name;
    }

    bool is_optimizable(vertex_t& v) {
      return graph[v].layer->is_optimizable();
    }

  private:
    NetworkGraph graph;

    int current_name_suffix = 1;

    // structure to build the graph from
    std::map<std::string, vertex_t> vertices;
    std::map<std::string, PtrOptimizer> optimizers;

    std::pair<std::string, std::shared_ptr<Loss>> name_loss;
    std::vector<std::string> input_vertices;

    std::deque<vertex_t> forward_order;
    bool inited = false;
  };

} // namespace EigenSinn