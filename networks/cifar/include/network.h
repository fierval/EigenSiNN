#pragma once

#pragma once

#include <layers/layer_base.hpp>
#include <optimizers/optimizer_base.hpp>
#include <layers/convolution.hpp>
#include <layers/flatten.hpp>
#include <layers/maxpooling.hpp>
#include <layers/linear.hpp>
#include <layers/relu.hpp>

using namespace Eigen;
using namespace EigenSinn;

struct NetworkNode {
  LayerBase* layer;
  OptimizerBase<float>* optimizer;
  NetworkNode() : layer(nullptr), optimizer(nullptr) {}

  ~NetworkNode() {
    if (layer) delete layer;
    if (optimizer) delete optimizer;
  }

  NetworkNode(NetworkNode&& other)  noexcept : NetworkNode() {
    layer = other.layer;
    optimizer = other.optimizer;

    other.layer = nullptr;
    other.optimizer = nullptr;
  }

  NetworkNode(LayerBase * _layer, OptimizerBase<float>* _optimizer) : layer(_layer), optimizer(_optimizer) {}
  NetworkNode(LayerBase * _layer) : layer(_layer), optimizer(nullptr) {}
};


typedef std::vector<NetworkNode> Network;

// assuming that the last layer of the network is Flatten, we can get the flattened dimension
inline int get_flat_dimension(const Network& network, const array<Index, 4>& input_dims) {
  Tensor<float, 4> input(input_dims);
  input.setZero();

  std::any tensor(input);

  for (const auto& n : network) {
    n.layer->forward(tensor);
    tensor = n.layer->get_output();
  }

  return from_any<float, 2>(tensor).dimension(1);
}

inline auto create_network(int batch_size, int num_classes, float learning_rate) {

  Network network;

  // push back rvalues so we don't have to invoke the copy constructor
  network.push_back(NetworkNode(new Conv2d<float>(array<Index, 4>{6, 3, 5, 5}), new Adam<float, 4>(learning_rate)));
  network.push_back(NetworkNode(new ReLU<float, 4>()));
  network.push_back(NetworkNode(new MaxPooling<float, 4>(array<Index, 2>{2, 2}, 2)));

  network.push_back(NetworkNode(new Conv2d<float>(array<Index, 4>{16, 6, 5, 5}), new Adam<float, 4>(learning_rate)));
  network.push_back(NetworkNode(new ReLU<float, 4>())); 
  network.push_back(NetworkNode(new MaxPooling<float, 4>(array<Index, 2>{2, 2}, 2)));

  network.push_back(NetworkNode(new Flatten<float>()));

  // get flat dimension by pushing a zero tensor through the network defined so far
  int flat_dim = get_flat_dimension(network, array<Index, 4>{1, 3, 32, 32});

  network.push_back(NetworkNode(new Linear<float>(batch_size, flat_dim, 120)));
  network.push_back(NetworkNode(new ReLU<float, 2>()));

  network.push_back(NetworkNode(new Linear<float>(batch_size, 120, 84)));
  network.push_back(NetworkNode(new ReLU<float, 2>()));

  // cross-entropy loss includes the softmax non-linearity
  network.push_back(NetworkNode(new Linear<float>(batch_size, 84, num_classes)));

  return network;
}

inline auto init_network(std::vector<NetworkNode>& network) {

  for (auto& node : network) {
    node.layer->init();
  }
}