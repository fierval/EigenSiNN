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
  Tensor<float, 4>  output;

  for (const auto& n : network) {
    n.layer->forward(tensor);
    tensor = n.layer->get_output();
    output = from_any<float, 4>(tensor);
  }

  auto dims = from_any<float, 4>(tensor).dimensions();

  return dims[1] * dims[2] * dims[3];
}

inline void init(const Network& network) {
  for (const auto& n : network) {
    n.layer->init();
  }
}

inline Tensor<float, 2> forward(const Network& network, const Tensor<float, 4>& input) {

  std::any tensor(input);

  Tensor<float, 2> output;

  for (const auto& n : network) {
    n.layer->forward(tensor);
    tensor = n.layer->get_output();
  }

  output = from_any<float, 2>(tensor);
  return output;
}

inline void backward(const Network& network, std::any& loss_derivative, std::any& input) {

  std::any next_level_grad = loss_derivative;

  // point at the "previous" layer
  auto reverse_input_it = network.rbegin() + 1;
  for (const auto& n : network) {
    // get previous layer output = input to this layer
    auto prev_layer = reverse_input_it != network.rend() ? reverse_input_it->layer->get_output() : input;

    n.layer->backward(prev_layer, next_level_grad);
    next_level_grad = n.layer->get_loss_by_input_derivative();
    reverse_input_it--;
  }
}

inline void optimizer(const Network& network) {

  for (auto optit = network.rbegin(); optit != network.rend(); optit++) {
    if (optit->optimizer == nullptr) {
      continue;
    }

    std::any weights, bias;
    auto layer = optit->layer;

    std::tie(weights, bias) = optit->optimizer->step(layer->get_weights(), layer->get_bias(), layer->get_loss_by_weights_derivative(), layer->get_loss_by_bias_derivative());
    layer->set_weights(weights);
    layer->set_bias(bias);
  }

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

  // get flat dimension by pushing a zero tensor through the network defined so far
  int flat_dim = get_flat_dimension(network, array<Index, 4>{1, 3, 32, 32});

  network.push_back(NetworkNode(new Flatten<float>()));


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