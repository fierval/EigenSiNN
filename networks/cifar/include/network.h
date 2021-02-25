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

namespace EigenSinn {
  template <typename Scalar, typename Device_ = ThreadPoolDevice >
  struct NetworkNode {
    std::unique_ptr<LayerBase<Scalar>> layer;
    std::unique_ptr<OptimizerBase<Scalar, Device_>> optimizer;

    NetworkNode(LayerBase<Scalar>* _layer, OptimizerBase<Scalar,Device_, 0>* _optimizer = nullptr) : layer(_layer), optimizer(_optimizer) {}

    NetworkNode(NetworkNode<Scalar, Device_>&& other)  noexcept {
      layer = std::move(other.layer);
      optimizer = std::move(other.optimizer);
    }

  };


  template<typename Scalar, Index Rank, typename Loss, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class NetBase {

    typedef std::vector<NetworkNode<Scalar, Device_>> Network;

  public:

    NetBase() : inited(false) {}

    inline virtual void forward() {

      if (!inited) {
        throw std::logic_error("Initialize network weights first");
      }

      for (auto it = network.begin() + 1; it != network.end(); it++) {
        it->layer->forward(*(it - 1)->layer);
      }
    }

    inline virtual void init() {
      for (int i = 0; i < network.size(); i++) {
        network[i].layer->init();
      }

      inited = true;
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
    template<typename Actual, Index OutputRank>
    inline void step(DeviceTensor<Scalar, Rank, Device_, Layout>& batch_tensor, DeviceTensor<Actual, OutputRank, Device_, Layout>& label_tensor) {

      // get the input
      dynamic_cast<Input<Scalar, Rank>*>(network[0].layer.get())->set_input(batch_tensor);

      // forward step
      forward();

      // loss step
      DeviceTensor<Scalar, OutputRank> output(network.rbegin()->layer->get_output());
      loss.step(output, label_tensor);

      // backward step
      backward(loss.get_loss_derivative_by_input());

      // optimizer steop
      optimizer();
    }

    inline void set_input(DeviceTensor<Scalar, Rank, Device_, Layout>& tensor) {

      auto inp_layer = dynamic_cast<Input<float, 4>*>(network[0].layer.get());
      inp_layer->set_input(tensor);
    }

    inline std::any get_output() {
      return network.rbegin()->layer.get()->get_output();
    }

    inline void add(NetworkNode<Scalar, Device_>* n) {
      network.push_back(n);
    }

    inline std::any get_loss() { return loss.get_output(); }

  protected:

    // assuming that the last layer of the network is Flatten, we can get the flattened dimension
    inline int get_flat_dimension(const array<Index, Rank>& input_dims) {

      // or it will throw during the forward pass
      init();

      DeviceTensor<Scalar, Rank, Device_, Layout> inp(input_dims);
      inp.setZero();

      dynamic_cast<Input<Scalar, Rank>*>(network[0].layer.get())->set_input(inp);

      forward();

      auto dims = DeviceTensor<Scalar, Rank>(network.rbegin()->layer->get_output()).dimensions();
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

    Network network;
    Loss loss;
    bool inited;
  };

  template<typename Device_ = ThreadPoolDevice>
  class Cifar10 : public NetBase<float, 4, CrossEntropyLoss<float, uint8_t, 2, Device_>, Device_> {

  public:
    Cifar10(array<Index, 4> input_dims, int num_classes, float learning_rate) {

      // push back rvalues so we don't have to invoke the copy constructor
      network.push_back(NetworkNode<float, Device_>(new Input<float, 4>));

      network.push_back(NetworkNode<float, Device_>(new Conv2d<float>(array<Index, 4>{6, 3, 5, 5}, Padding2D{ 0, 0 }, 1), get_optimizer<4>(learning_rate)));
      network.push_back(NetworkNode<float, Device_>(new ReLU<float, 4>));
      network.push_back(NetworkNode<float, Device_>(new MaxPooling<float, 4>(array<Index, 2>{2, 2}, 2)));

      network.push_back(NetworkNode<float, Device_>(new Conv2d<float>(array<Index, 4>{16, 6, 5, 5}, Padding2D{ 0, 0 }, 1), get_optimizer<4>(learning_rate)));
      network.push_back(NetworkNode<float, Device_>(new ReLU<float, 4>));
      network.push_back(NetworkNode<float, Device_>(new MaxPooling<float, 4>(array<Index, 2>{2, 2}, 2)));

      // get flat dimension by pushing a zero tensor through the network defined so far
      int flat_dim = get_flat_dimension(array<Index, 4>{1, 3, 32, 32});

      network.push_back(NetworkNode<float, Device_>(new Flatten<float>));


      network.push_back(NetworkNode<float, Device_>(new Linear<float>(flat_dim, 120), get_optimizer<2>(learning_rate)));
      network.push_back(NetworkNode<float, Device_>(new ReLU<float, 2>));

      network.push_back(NetworkNode<float, Device_>(new Linear<float>(120, 84), get_optimizer<2>(learning_rate)));
      network.push_back(NetworkNode<float, Device_>(new ReLU<float, 2>));

      // cross-entropy loss includes the softmax non-linearity
      network.push_back(NetworkNode<float, Device_>(new Linear<float>(84, num_classes), get_optimizer<2>(learning_rate)));
    }

  protected:
    template <Index Rank>
    inline auto get_optimizer(float learning_rate) {
      return dynamic_cast<OptimizerBase<float>*>(new SGD<float, Rank>(learning_rate, 0, false));
    }
  };
}