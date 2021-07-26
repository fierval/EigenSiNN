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

  template <typename Scalar, typename Actual, typename Loss, typename Device_>
  class NetworkBase {

  public:

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
    typedef std::shared_ptr<OptimizerBase<Scalar, Device_, RowMajor>> PtrOptimizer;
    typedef PtrTensorAdapter<Scalar, Device_> PtrTensor;
    typedef PtrTensorAdapter<Actual, Device_> PtrTensorActual;
    typedef std::vector<PtrTensorAdapter<Scalar, Device_>> TensorVector;
    typedef std::shared_ptr<Loss> PtrLoss;


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

    // forward step with the map of input names -> inputs and logit names -> labels
    void forward(std::unordered_map<std::string, PtrTensor> tensors, std::unordered_map<std::string, PtrTensorActual> labels) {

      if (tensors.empty() || labels.empty()) {
        throw std::invalid_argument("inputs and labels must be non-empty");
      }

      // we should have topologically sorted the graph
      assert(forward_order.size() > 0);

      if (!inited) {
        init();
      }

      set_input(tensors);

      for (auto& v : forward_order) {

        TensorVector inputs = collect_inputs(v);
        auto& layer_name = graph[v].layer->get_layer_name();

        // input layer. all inputs are set by the set_input call
        if (inputs.empty()) { continue; }

        graph[v].layer->forward(inputs);

        // if we have reached a terminal node - compute loss
        if (name_loss.count(layer_name) > 0) {
          PtrLoss& loss = name_loss[layer_name];

          loss->step(graph[v].layer->get_output(), labels[layer_name]);
        }
      }
    }

    void forward(PtrTensor& tensor, PtrTensorActual& label) {

      std::unordered_map<std::string, PtrTensor> inputs;
      std::unordered_map<std::string, PtrTensorActual> labels;

      inputs.insert(std::make_pair(input_vertices[0], tensor));
      labels.insert(std::make_pair(name_loss.begin()->first, label));

      forward(inputs, labels);
    }

    std::vector<std::string>& get_input_names() {
      return input_vertices;
    }

    void backward() {

      assert(forward_order.size() > 0);

      OutEdgeIter out_layer_edge, out_layer_edge_end;

      // vertex names when we need to retrieve losses
      vertex_name names = get(boost::vertex_name, graph);

      // iterate over sorted vertices in reverse order
      for (auto it = forward_order.rbegin(); it != forward_order.rend(); it++) {
        vertex_t v = *it;

        TensorVector prev_layers = collect_inputs(v);

        // reached input layer
        if (prev_layers.empty()) { continue; }

        // TODO: for now should only be a single one
        boost::tie(out_layer_edge, out_layer_edge_end) = boost::out_edges(v, graph);

        PtrTensor incoming_derivative;

        // No outputs meaning we need to attach a loss
        // we have already computed the loss and its derivative
        // in the "forward" step
        if (out_layer_edge == out_layer_edge_end) {
          PtrLoss& loss = name_loss[names[v]];
          incoming_derivative = loss->get_loss_derivative_by_input();
        }
        else {
          vertex_t target = boost::target(*out_layer_edge, graph);
          incoming_derivative = graph[target].layer->get_loss_by_input_derivative();
        }

        graph[v].layer->backward(prev_layers, incoming_derivative);
      }

    }

    void optimize() {
      for (auto& p : optimizers) {
        vertex_t v = vertices[p.first];
        p.second->step(*graph[v].layer);
      }
    }

    void step(std::unordered_map<std::string, PtrTensor> tensors, std::unordered_map<std::string, PtrTensorActual> labels) {
      forward(tensors, labels);
      backward();
      optimize();
    }

    void step(PtrTensor& tensor, PtrTensorActual& label) {

      std::unordered_map<std::string, PtrTensor> inputs;
      std::unordered_map<std::string, PtrTensorActual> labels;

      inputs.insert(std::make_pair(input_vertices[0], tensor));
      labels.insert(std::make_pair(name_loss.begin()->first, label));

      step(inputs, labels);
    }

    // exploratory functions
    // ======================


    std::vector<std::string> get_layer_names() {

      std::vector<std::string> out;
      std::transform(vertices.begin(), vertices.end(), std::back_inserter(out),
        [](std::pair<std::string, vertex_t>& p) {return p.first; });

      return out;
    }

    PtrLayer get_layer(const std::string& name) {

      if (vertices.count(name) == 0) {
        throw std::invalid_argument("Invlaid layer name");
      }

      return graph[vertices[name]].layer;

    }
  protected:

    // single input network
    void set_input(PtrTensor& input) {

      assert(input_vertices.size() == 1);

      std::unordered_map<std::string, PtrTensor> inputs;
      inputs.insert(std::make_pair(input_vertices[0], input));
      set_input(inputs);
    }

    void set_input(std::unordered_map<std::string, PtrTensor>& inputs) {

      assert(inputs.size() > 0);
      std::for_each(inputs.begin(), inputs.end(), [&](std::pair<std::string, PtrTensor> p) {
        PtrLayer layer = graph[vertices[p.first]].layer;
        Input<Scalar, Device_>* input_layer = dynamic_cast<Input<Scalar, Device_>*>(layer.get());
        input_layer->set_input(p.second);
        });
    }

    // add loss function. There may be several output nodes
    // so we may call this several times
    void add_loss(const std::string& logit) {

      add_loss(std::vector<std::string>{ logit });
    }

    void add_loss(const std::vector<std::string>& logits) {
      this->logits = logits;
      add_loss();
    }

    void add_loss() {
      assert(!logits.empty());
      for (auto& l : logits) {
        name_loss.insert(std::make_pair(l, std::make_shared<Loss>()));
      }
    }

    template <typename... Args>
    void set_optimizer(const Optimizers optimizer_type, float lr, Args... args) {

      this->optimizer_type = optimizer_type;
      this->optimizer_args = { args... };
      this->lr = lr;
    }

    // optimizers have different parameters
    // either pass just lr

    virtual void add_optimizer(vertex_t v) {

      // set the optimizer first
      assert(optimizer_type != Optimizers::None);

      PtrOptimizer base_optimizer;
      // rank = 0 for layers that don't need an optimizer
      Index rank = get_optimizer_rank(v);
      switch (get_optimizer_rank(v))
      {
      case 1:
        base_optimizer = create_optimizer<1>();
        break;
      case 2:
        base_optimizer = create_optimizer<2>();
        break;
      case 4:
        base_optimizer = create_optimizer<4>();
        break;
      default:
        return;
      }

      optimizers.insert(std::make_pair(graph[v].layer->get_layer_name(), base_optimizer));
    }

    // TODO: This is SO ugly!!! Due to the stupid optimizer rank
    template<Index Rank>
    PtrOptimizer create_optimizer() {

      PtrOptimizer base_optimizer;

      switch (optimizer_type) {

      case Optimizers::Adam:
        base_optimizer.reset(new Adam<Scalar, Rank, Device_>(lr, optimizer_args));
        break;

      case Optimizers::SGD:
        base_optimizer.reset(new SGD<Scalar, Rank, Device_>(lr, optimizer_args));
        break;

      default:
        throw std::invalid_argument("Undknown uptimizer");
      }

      return base_optimizer;
    }

    // add optimizers where they belong and produce forward traversal
    template<typename... Args>
    void compile(Optimizers optimizer_name, float lr, Args... args) {
      
      assert(forward_order.size() == 0);
      set_optimizer(optimizer_name, lr, args...);

      add_optimizers();
      compile();
    }

    void add_optimizers() {
      // find opimizable layers and hang an optimizer on them
      std::for_each(vertices.begin(), vertices.end(), [&](std::pair<std::string, vertex_t> p) {
        add_optimizer(p.second);
        });

    }
    void compile() {
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
      if (layer->get_layer_name().empty()) {
        layer->set_layer_name(get_layer_name(layer->get_op_name()));
      }
      return layer->get_layer_name();
    }

    std::string add(Input<Scalar, Device_>* inp_layer) {

      std::string& inp_layer_name = inp_layer->get_layer_name();

      // we may been here before, but not necessarily!
      if (!inp_layer_name.empty() && vertices.count(inp_layer_name) > 0) {
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
    std::string add(const std::string& input_name, LayerBase<Scalar, Device_>* layer) {

      // we need to have the input already inserted
      assert(vertices.count(input_name) > 0);

      std::string& layer_name = layer->get_layer_name();
      vertex_t v_out, v_in;

      // name the current vertex if it doesn't yet have a name
      if (layer_name.empty() || vertices.count(layer_name) == 0) {
        layer_name = name_the_layer(layer);

        v_out = boost::add_vertex(VertexProperty(layer_name), graph);
        graph[v_out].layer.reset(layer);

        vertices.insert(std::make_pair(layer_name, v_out));
      }

      v_out = vertices[layer_name];
      v_in = vertices[input_name];
      boost::add_edge(v_in, v_out, graph);

      return layer_name;
    }

    Index get_optimizer_rank(vertex_t& v) {
      return graph[v].layer->get_optimizer_rank();
    }

    // given a vertex return all vertices feeding into it
    TensorVector collect_inputs(vertex_t& v) {

      TensorVector inputs;
      InEdgeIter in, in_end;

      for (boost::tie(in, in_end) = boost::in_edges(v, graph); in != in_end; in++) {

        vertex_t inp = boost::source(*in, graph);
        inputs.push_back(graph[inp].layer->get_output());
      }

      return inputs;
    }

    void clear() {
      vertices.clear();
      optimizers.clear();
      name_loss.clear();
      forward_order.clear();
      graph.clear();
      input_vertices.clear();
      logits.clear();
    }

    // prevent direct instantiation
    virtual const std::string& model_name() = 0;

    NetworkGraph graph;

    int current_name_suffix = 1;

    // structure to build the graph from
    std::unordered_map<std::string, vertex_t> vertices;
    std::unordered_map<std::string, PtrOptimizer> optimizers;

    // TODO: a single output NN.
    std::unordered_map<std::string, PtrLoss> name_loss;
    std::vector<std::string> input_vertices;
    std::vector<std::string> logits;

    std::deque<vertex_t> forward_order;
    bool inited = false;

    // should be set by the implementation
    Optimizers optimizer_type = Optimizers::None;
    std::vector<float> optimizer_args;
    float lr;
  };

} // namespace EigenSinn