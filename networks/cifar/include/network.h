#pragma once

#pragma once

#include <layers/layer_base.hpp>
#include <optimizers/sgd.hpp>
#include <layers/convolution.hpp>
#include <layers/flatten.hpp>
#include <layers/maxpooling.hpp>
#include <layers/linear.hpp>
#include <layers/relu.hpp>

#include "helpers.h"

using namespace Eigen;
using namespace EigenSinn;

template <typename Device_= DefaultDevice>
struct NetworkNode {
  LayerBase<Device_> * layer;
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

  NetworkNode(LayerBase<Device_>* _layer, OptimizerBase<float>* _optimizer) : layer(_layer), optimizer(_optimizer) {}
  NetworkNode(LayerBase<Device_>* _layer) : layer(_layer), optimizer(nullptr) {}
};


typedef std::vector<NetworkNode<ThreadPoolDevice>> Network;

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

inline void init(const Network& network, bool debug = false) {

  const std::string base_path = "C:\\git\\NN\\pytorchTutorial\\layer_";
  int j = 0;
  for (int i = 0; i < network.size(); i++) {

    if (debug && (i == 0 || i == 3 || i == 7 || i == 9 || i == 11)) {
      std::string full_path = base_path + std::to_string(j);
      j++;

      if (i < 7) {
        Tensor<float, 4> init_tensor = read_tensor_csv<float, 4>(full_path);
        dynamic_cast<Conv2d<float>*>(network[i].layer)->init(init_tensor);
      }
      else {
        Tensor<float, 2> init_tensor = read_tensor_csv<float, 2>(full_path);
        // pytorch returns a transposed weight matrix
        Tensor<float, 2> transposed = init_tensor.shuffle(array<Index, 2>{1, 0});

        dynamic_cast<Linear<float>*>(network[i].layer)->init(transposed);
      }

      continue;
    }
    network[i].layer->init();
  }
}

inline Tensor<float, 2> forward(const Network& network, std::any& input) {

  std::any tensor = input;

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
  for (auto it = network.rbegin(); it != network.rend(); it++) {

    // get previous layer output = input to this layer
    auto prev_layer = reverse_input_it != network.rend() ? reverse_input_it->layer->get_output() : input;

    // backward pass through the current layer
    it->layer->backward(prev_layer, next_level_grad);
    next_level_grad = it->layer->get_loss_by_input_derivative();
    
    if (reverse_input_it != network.rend()) {
      reverse_input_it++;
    }
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

inline auto create_network(int num_classes, float learning_rate, Dispatcher<ThreadPoolDevice>& device) {

  Network network;

  // push back rvalues so we don't have to invoke the copy constructor
  network.push_back(NetworkNode<ThreadPoolDevice>(new Conv2d<float, ThreadPoolDevice>(array<Index, 4>{6, 3, 5, 5}, Padding2D{ 0, 0 }, 1, device), new SGD<float, 4>(learning_rate)));
  network.push_back(NetworkNode<ThreadPoolDevice>(new ReLU<float, 4, ThreadPoolDevice>(device)));
  network.push_back(NetworkNode<ThreadPoolDevice>(new MaxPooling<float, 4, ThreadPoolDevice>(array<Index, 2>{2, 2}, 2, device)));

  network.push_back(NetworkNode<ThreadPoolDevice>(new Conv2d<float, ThreadPoolDevice>(array<Index, 4>{16, 6, 5, 5}, Padding2D{ 0, 0 }, 1, device), new SGD<float, 4>(learning_rate)));
  network.push_back(NetworkNode<ThreadPoolDevice>(new ReLU<float, 4, ThreadPoolDevice>(device)));
  network.push_back(NetworkNode<ThreadPoolDevice>(new MaxPooling<float, 4, ThreadPoolDevice>(array<Index, 2>{2, 2}, 2, device)));

  // get flat dimension by pushing a zero tensor through the network defined so far
  int flat_dim = get_flat_dimension(network, array<Index, 4>{1, 3, 32, 32});

  network.push_back(NetworkNode<ThreadPoolDevice>(new Flatten<float, ThreadPoolDevice>(device)));


  network.push_back(NetworkNode<ThreadPoolDevice>(new Linear<float, ThreadPoolDevice>(flat_dim, 120, device), new SGD<float, 2>(learning_rate)));
  network.push_back(NetworkNode<ThreadPoolDevice>(new ReLU<float, 2, ThreadPoolDevice>(device)));

  network.push_back(NetworkNode<ThreadPoolDevice>(new Linear<float, ThreadPoolDevice>(120, 84, device), new SGD<float, 2>(learning_rate)));
  network.push_back(NetworkNode<ThreadPoolDevice>(new ReLU<float, 2, ThreadPoolDevice>(device)));

  // cross-entropy loss includes the softmax non-linearity
  network.push_back(NetworkNode<ThreadPoolDevice>(new Linear<float, ThreadPoolDevice>(84, num_classes, device), new SGD<float, 2>(learning_rate)));

  return network;
}
