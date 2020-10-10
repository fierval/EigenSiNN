#pragma once

#pragma once

#include <layers/layer_base.hpp>
#include <optimizers/sgd.hpp>
#include <layers/convolution.hpp>
#include <layers/flatten.hpp>
#include <layers/maxpooling.hpp>
#include <layers/linear.hpp>
#include <layers/relu.hpp>
#include <layers/input.hpp>

#include "helpers.h"

using namespace Eigen;
using namespace EigenSinn;

template <typename Device_= DefaultDevice>
struct NetworkNode {
  LayerBase<float, Device_> * layer;
  OptimizerBase<float, Device_>* optimizer;
  NetworkNode() : layer(nullptr), optimizer(nullptr) {}

  ~NetworkNode() {
    if (layer) delete layer;
    if (optimizer) delete optimizer;
  }

  NetworkNode(NetworkNode<Device_>&& other)  noexcept : NetworkNode<Device_>() {
    layer = other.layer;
    optimizer = other.optimizer;

    other.layer = nullptr;
    other.optimizer = nullptr;
  }

  NetworkNode(LayerBase<float, Device_>* _layer, OptimizerBase<float>* _optimizer) : layer(_layer), optimizer(_optimizer) {}
  NetworkNode(LayerBase<float, Device_>* _layer) : layer(_layer), optimizer(nullptr) {}
};


typedef std::vector<NetworkNode<ThreadPoolDevice>> Network;

// assuming that the last layer of the network is Flatten, we can get the flattened dimension
inline int get_flat_dimension(const Network& network, const array<Index, 4>& input_dims) {
  Tensor<float, 4> inp(input_dims);
  inp.setZero();

  dynamic_cast<Input<float, 4>*>(network[0].layer)->set_input(inp.data());

  forward(network);

  auto dims = network.rbegin()->layer->get_out_dims();

  return std::accumulate(dims.begin() +1, dims.end(), 1, std::multiplies<int>());
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

inline void forward(const Network& network) {

  for (auto it = network.begin() + 1; it != network.end(); it++) {
    it->layer->forward(*(it - 1)->layer);
  }
}

inline void backward(const Network& network, float * loss_derivative) {

  for (auto rit = network.rbegin(); rit != network.rend() - 1; rit++) {

    if (rit == network.rbegin()) {
      rit->layer->backward(*(rit + 1)->layer, loss_derivative);
    }
    else {
      rit->layer->backward(*(rit + 1)->layer, *(rit - 1)->layer);
    }
  }
}

inline void optimizer(const Network& network) {

  for (auto optit = network.rbegin(); optit != network.rend(); optit++) {
    if (optit->optimizer == nullptr) {
      continue;
    }

    auto layer = optit->layer;

    optit->optimizer->step(*(optit->layer));
  }

}

inline auto create_network(array<Index, 4> input_dims, int num_classes, float learning_rate, Dispatcher<ThreadPoolDevice>& device) {

  Network network;

  // push back rvalues so we don't have to invoke the copy constructor
  network.push_back(NetworkNode<ThreadPoolDevice>(new Input<float, 4, ThreadPoolDevice>(input_dims, device)));
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
