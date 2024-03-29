#pragma once

#pragma once

#include <optimizers/sgd.hpp>
#include <layers/layer_base.hpp>
#include <layers/convolution.hpp>
#include <layers/flatten.hpp>
#include <layers/maxpooling.hpp>
#include <layers/linear.hpp>
#include <layers/relu.hpp>
#include <layers/input.hpp>

#include <onnx/loader.h>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  struct NetworkNode {
    std::shared_ptr<LayerBase<Scalar, Device_>> layer;
    std::shared_ptr<OptimizerBase<Scalar, Device_, Layout>> optimizer;

    NetworkNode(LayerBase<Scalar, Device_>* _layer, OptimizerBase<Scalar, Device_, Layout>* _optimizer = nullptr) : layer(_layer), optimizer(_optimizer) {}

    NetworkNode(NetworkNode<Scalar, Device_>&& other)  noexcept {
      layer = std::move(other.layer);
      optimizer = std::move(other.optimizer);
    }

  };


  template<typename Scalar, Index Rank, typename Loss, typename Device_ = ThreadPoolDevice, int Layout = RowMajor, bool CuDnn = false>
  class NetBase {

    typedef std::vector<NetworkNode<Scalar, Device_>> Network;
    typedef typename OnnxLoader<Scalar, Device_>::PtrLayer PtrLayer;

  public:

    NetBase() : inited(false) {}

    // CUDA compiler doesn't understand STL containers, so do it in plain-old loop
    inline virtual void forward() {

      if (!inited) {
        throw std::logic_error("Initialize network weights first");
      }

      for (int i = 1; i < network.size(); i++) {
        network[i].layer->forward(network[i - 1].layer->get_output());
      }
    }

    inline virtual void init() {
      for (int i = 0; i < network.size(); i++) {
        network[i].layer->init();
      }

      inited = true;
    }

    inline virtual void backward(const PtrTensorAdaptor<Scalar, Device_>& loss_derivative) {

      for (size_t i = network.size() - 1; i > 0; i--) {

        if (i == network.size() - 1) {
          network[i].layer->backward(network[i - 1].layer->get_output(), loss_derivative);
          continue;
        }
        network[i].layer->backward(network[i - 1].layer->get_output(), network[i + 1].layer->get_loss_by_input_derivative());
      }
    }

    inline virtual void optimizer() {

      for (size_t i = network.size() - 1; i > 0; i--) {

        if (!network[i].optimizer) {
          continue;
        }

        network[i].optimizer->step(*(network[i].layer));
      }
    }

    // Run the step of back-propagation
    template<typename Actual, Index OutputRank>
    inline void step(DeviceTensor<Scalar, Rank, Device_, Layout>& batch_tensor, DeviceTensor<Actual, OutputRank, Device_, Layout>& label_tensor) {

      // get the input
      set_input(batch_tensor);

      // forward step
      forward();

      // loss step
      DeviceTensor<Scalar, OutputRank, Device_, Layout> output(network.rbegin()->layer->get_output());
      loss.step(output.raw(), label_tensor.raw());

      // backward step
      backward(loss.get_loss_derivative_by_input());

      // optimizer steop
      optimizer();
    }

    inline PtrTensorAdaptor<Scalar, Device_> get_output() {
      return network.rbegin()->layer.get()->get_output();
    }

    inline void add(LayerBase<Scalar, Device_>* n, OptimizerBase<Scalar, Device_, Layout>* opt = nullptr) {
      n->set_cudnn(CuDnn);
      network.push_back(NetworkNode<Scalar, Device_>(n, opt));
    }

    inline void clear() {
      network.clear();
    }

    inline Scalar get_loss() { return loss.get_output(); }

    inline void save_to_onnx(EigenModel& model) {

      auto * inp_layer = dynamic_cast<Input<Scalar,Device_>*>(network[0].layer.get());

      std::string output_name = "input.1";
      auto input_dims = inp_layer->get_dims();

      // when saving for inference we
      // set the 0th dimension to 1 
      // as inference is mostly done 1 image/time
      if (model.is_inference()) {
        input_dims[0] = 1;
      }
      // 1. Input layer
      model.add_input(output_name, input_dims, onnx_data_type_from_scalar<Scalar>());

      // 2. Serialize each layer
      for (int i = 1; i < network.size(); i++) {
        auto& node = network[i];
        LayerBase<Scalar, Device_>* layer = node.layer.get();
        output_name = layer->add_onnx_node(model, output_name);
      }
      
      // 3. Output layer
      model.add_output(output_name, network[network.size() - 1].layer->onnx_out_dims(), onnx_data_type_from_scalar<Scalar>());
    }

    // create network by loading it from a byte string
    inline void load_from_onnx(const std::string& data) {

      EigenModel model(data);

      // create input layer
      add(new Input<float, Device_>);

      onnx::GraphProto* graph = model.get_graph();
      OnnxLoader<Scalar, Device_> onnx_layers(model);

      for (auto& node : graph->node()) {
        StringVector inputs;
        std::string output;
        LayerBase<Scalar, Device_> * layer;
        std::tie(inputs, output, layer) = onnx_layers.create_from_node(node);

        add(layer);
        layer->set_cudnn(CuDnn);
      }

    }

    inline void set_input(DeviceTensor<Scalar, Rank, Device_, Layout>& tensor) {

      auto inp_layer = dynamic_cast<Input<Scalar, Device_>*>(network[0].layer.get());
      inp_layer->set_input(tensor);
    }

  protected:

    // assuming that the last layer of the network is Flatten, we can get the flattened dimension
    inline int get_flat_dimension(const array<Index, Rank>& input_dims) {

      // or it will throw during the forward pass
      init();

      DeviceTensor<Scalar, Rank, Device_, Layout> inp(input_dims);
      inp.setZero();

      dynamic_cast<Input<Scalar, Device_>*>(network[0].layer.get())->set_input(inp);

      forward();

      auto dims = DeviceTensor<Scalar, Rank, Device_, Layout>(network.rbegin()->layer->get_output()).dimensions();
      int res = 1;

      // no STL for CUDA, can't use std::accumulate
      // skip the first - batch dimension
      for (int i = 1; i < dims.size(); i++) {
        res *= dims[i];
      }

      // we will need to initialize remaining layers
      inited = false;
      return res;
    }

    template <Index Rank>
    inline OptimizerBase<float, Device_, Layout>* get_optimizer(float learning_rate) {
      return dynamic_cast<OptimizerBase<float, Device_, Layout>*>(new SGD<float, Rank, Device_, Layout>(learning_rate, 0, false));
    }

    Network network;
    Loss loss;
    bool inited;
  };
}