#pragma once

#include <layers/layer_base.hpp>
#include <optimizers/optimizer_base.hpp>

using namespace Eigen;
using namespace EigenSinn;

struct NetworkNode {
  LayerBase * layer;
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

  NetworkNode(const NetworkNode& other) : NetworkNode() {
    layer = other.layer;
    optimizer = other.optimizer;

  }

};


inline auto create_network(int batch_size, int input_size, int hidden_size, int num_classes, float learning_rate) {

  std::vector<NetworkNode> network;

  // create the network
  NetworkNode l1, relu1, l2;

  // layers
  l1.layer = new Linear<float>(batch_size, input_size, hidden_size);
  relu1.layer = new ReLU<float, 2>();
  l2.layer = new Linear<float>(batch_size, hidden_size, num_classes);

  // optimizers
  l1.optimizer = new Adam<float, 2>(learning_rate);
  l2.optimizer = new Adam<float, 2>(learning_rate);
  
  network.push_back(l1);
  network.push_back(relu1);
  network.push_back(l2);

  return network;
}