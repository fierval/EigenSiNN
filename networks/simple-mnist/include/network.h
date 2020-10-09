#pragma once

#include <layers/layer_base.hpp>
#include <optimizers/optimizer_base.hpp>
#include <layers/linear.hpp>
#include <layers/relu.hpp>
#include <layers/input.hpp>

using namespace Eigen;
using namespace EigenSinn;

struct NetworkNode {
  LayerBase<float> * layer;
  OptimizerBase<float> * optimizer;
  NetworkNode() : layer(nullptr), optimizer(nullptr) {}

  ~NetworkNode() {
    if (layer) delete layer;
    if (optimizer) delete optimizer;
  }

  NetworkNode(NetworkNode&& other)  noexcept : NetworkNode(){
    layer = other.layer;
    optimizer = other.optimizer;

    other.layer = nullptr;
    other.optimizer = nullptr;
  }

  NetworkNode(LayerBase<float>* _layer, OptimizerBase<float> * _optimizer) : layer(_layer), optimizer(_optimizer) {}
};


inline auto create_network(int batch_size, int input_size, int hidden_size, int num_classes, float learning_rate) {

  std::vector<NetworkNode> network;

  // push back rvalues so we don't have to invoke the copy constructor
  network.emplace_back(NetworkNode(new Input<float, 2>(array<Index, 2>{batch_size, input_size}), nullptr));
  network.emplace_back(NetworkNode(new Linear<float>(input_size, hidden_size), new Adam<float, 2>(learning_rate)));
  network.emplace_back(NetworkNode(new ReLU<float, 2>(), nullptr));
  network.emplace_back(NetworkNode(new Linear<float>(hidden_size, num_classes), new Adam<float, 2>(learning_rate)));

  return network;
}

inline auto init_network(std::vector<NetworkNode>& network) {

  for (auto& node : network) {
    node.layer->init();
  }
}