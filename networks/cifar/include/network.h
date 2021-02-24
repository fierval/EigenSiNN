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

#include "helpers.h"

using namespace Eigen;
using namespace EigenSinn;

namespace network {
  template <typename Scalar, typename Device_ = ThreadPoolDevice >
  struct NetworkNode {
    std::unique_ptr<LayerBase<Scalar>> layer;
    std::unique_ptr<OptimizerBase<Scalar, Device_>> optimizer;

    NetworkNode() : layer(nullptr), optimizer(nullptr) {}

    NetworkNode(NetworkNode<Device_>&& other)  noexcept : NetworkNode<Device_>() {
      layer = std::move(other.layer);
      optimizer = std::move(other.optimizer);
    }

    NetworkNode(LayerBase<Scalar>* _layer, OptimizerBase<Scalar, 0, Device_>* _optimizer) : layer(_layer), optimizer(_optimizer) {}
    
  };


  template< template<typename Scalar, typename Actual, Index Rank = 2, typename Device_ = ThreadPoolDevice, int Layout = ColMajor> class Loss>
  class NetBase {

    typedef std::vector<NetworkNode<Device_>> Network;

  public:
    inline virtual void forward() {
      for (auto it = network.begin() + 1; it != network.end(); it++) {
        it->layer->forward(*(it - 1)->layer);
      }
    }

    inline virtual void init() {
      for (int i = 0; i < network.size(); i++) {
        network[i].layer->init();
      }
    }

    inline virtual void backward(std::any loss_derivative) {

      for (auto rit = network.rbegin(); rit != network.rend() - 1; rit++) {

        if (rit == network.rbegin()) {
          rit->layer->backward(*(rit + 1)->layer, loss_derivative);
        }
        else {
          rit->layer->backward(*(rit + 1)->layer, (rit - 1)->layer->get_loss_by_input_derivative());
        }
      }
    }

    inline virtual void optimizer() {

      static std::any weights_any, bias_any;
      for (auto optit = network.rbegin(); optit != network.rend(); optit++) {
        if (optit->optimizer == nullptr) {
          continue;
        }

        std::tie(weights_any, bias_any) = optit->optimizer->step(*(optit->layer));
        optit->layer->set_weights(weights_any);
        optit->layer->set_bias(bias_any);
      }
    }

    // Run the step of back-propagation
    inline virtual void step(DeviceTensor<Scalar, Rank, Device_, Layout>& batch_tensor, DeviceTensor<Actual, Rank, Device_, Layout>& label_tensor) {

      // get the input
      dynamic_cast<Input<float, 4>*>(network[0].layer)->set_input(batch_tensor);

      // forward step
      forward();

      // loss step
      DeviceTensor<float, 2> output(network.rbegin()->layer->get_output());
      loss.step(output, label_tensor);

      // backward step
      backward(loss.get_loss_derivative_by_input());

      // optimizer steop
      optimizer();
    }

    inline void add(NetworkNode* n) {
      network.push_back(n);
    }

  protected:

    // assuming that the last layer of the network is Flatten, we can get the flattened dimension
    template<Index Rank>
    inline int get_flat_dimension(const Network& network, const array<Index, Rank>& input_dims) {
      Tensor<float, Rank> inp(input_dims);
      inp.setZero();

      dynamic_cast<Input<float, Rank>*>(network[0].layer)->set_input(inp);

      forward();

      auto dims = DeviceTensor<float, Rank>(network.rbegin()->layer->get_output()).dimensions();
      int res = 1;

      // no STL for CUDA, can't use std::accumulate
      // skip the first - batch dimension
      for (int i = 1; i < dims.size(); i++) {
        res *= dims[i];
      }
      return res;
    }

    Network network;
    Loss loss;

  };

  template<typename Device_ = ThreadPoolDevice>
  class Cifar10 : public NetBase<Device_> {

    inline void create_network(array<Index, 4> input_dims, int num_classes, float learning_rate) {

      Network network;

      // push back rvalues so we don't have to invoke the copy constructor
      network.push_back(NetworkNode<Device_>(new Input<float, 4>));

      network.push_back(NetworkNode<Device_>(new Conv2d<float>(array<Index, 4>{6, 3, 5, 5}, Padding2D{ 0, 0 }, 1), get_optimizer<4>(learning_rate)));
      network.push_back(NetworkNode<Device_>(new ReLU<float, 4>));
      network.push_back(NetworkNode<Device_>(new MaxPooling<float, 4>(array<Index, 2>{2, 2}, 2)));

      network.push_back(NetworkNode<Device_>(new Conv2d<float>(array<Index, 4>{16, 6, 5, 5}, Padding2D{ 0, 0 }, 1), get_optimizer<4>(learning_rate)));
      network.push_back(NetworkNode<Device_>(new ReLU<float, 4>));
      network.push_back(NetworkNode<Device_>(new MaxPooling<float, 4>(array<Index, 2>{2, 2}, 2)));

      // get flat dimension by pushing a zero tensor through the network defined so far
      int flat_dim = get_flat_dimension(network, array<Index, 4>{1, 3, 32, 32});

      network.push_back(NetworkNode<Device_>(new Flatten<float>));


      network.push_back(NetworkNode<Device_>(new Linear<float>(flat_dim, 120), get_optimizer<2>(learning_rate)));
      network.push_back(NetworkNode<Device_>(new ReLU<float, 2>));

      network.push_back(NetworkNode<Device_>(new Linear<float>(120, 84), get_optimizer<2>(learning_rate)));
      network.push_back(NetworkNode<Device_>(new ReLU<float, 2>));

      // cross-entropy loss includes the softmax non-linearity
      network.push_back(NetworkNode<Device_>(new Linear<float>(84, num_classes), get_optimizer<2>(learning_rate)));
    }

  protected:
    template <Index Rank>
    inline auto get_optimizer(float learning_rate) {
      return dynamic_cast<OptimizerBase<float>*>(new SGD<float, Rank>(learning_rate, 0, false));
    }
  };
}