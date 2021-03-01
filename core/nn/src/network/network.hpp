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
    std::unique_ptr<LayerBase<Scalar, Device_>> layer;
    std::unique_ptr<OptimizerBase<Scalar, Device_>> optimizer;

    NetworkNode(LayerBase<Scalar, Device_>* _layer, OptimizerBase<Scalar, Device_, 0>* _optimizer = nullptr) : layer(_layer), optimizer(_optimizer) {}

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

    // CUDA compiler doesn't understand STL containers, so do it in plain-old loop
    inline virtual void forward() {

      if (!inited) {
        throw std::logic_error("Initialize network weights first");
      }

      for (int i = 1; i < network.size(); i++) {
        network[i].layer->forward(*network[i - 1].layer);
      }
    }

    inline virtual void init() {
      for (int i = 0; i < network.size(); i++) {
        network[i].layer->init();
      }

      inited = true;
    }

    inline virtual void backward(PtrTensorAdapter<Scalar, Device_> loss_derivative) {

      PtrTensorAdapter<Scalar, Device_> derivative = loss_derivative;
      for (size_t i = network.size() - 1; i > 0; i--) {

        network[i].layer->backward(*network[i - 1].layer, derivative);
        derivative = network[i].layer->get_loss_by_input_derivative();
      }
    }

    inline virtual void optimizer() {

      static PtrTensorAdapter<Scalar, Device_> weights_any, bias_any;
      for (size_t i = network.size() - 1; i > 0; i--) {

        if (!network[i].optimizer) {
          continue;
        }

        std::tie(weights_any, bias_any) = network[i].optimizer->step(*(network[i].layer));
        network[i].layer->set_weights(weights_any);
        network[i].layer->set_bias(bias_any);
      }
    }

    // Run the step of back-propagation
    template<typename Actual, Index OutputRank>
    inline void step(DeviceTensor<Scalar, Rank, Device_, Layout>& batch_tensor, DeviceTensor<Actual, OutputRank, Device_, Layout>& label_tensor) {

      // get the input
      dynamic_cast<Input<Scalar, Rank, Device_, Layout>*>(network[0].layer.get())->set_input(batch_tensor);

      // forward step
      forward();

      // loss step
      DeviceTensor<Scalar, OutputRank, Device_, Layout> output(network.rbegin()->layer->get_output());
      loss.step(output, label_tensor);

      // backward step
      backward(loss.get_loss_derivative_by_input());

      // optimizer steop
      optimizer();
    }

    inline void set_input(DeviceTensor<Scalar, Rank, Device_, Layout>& tensor) {

      auto inp_layer = dynamic_cast<Input<float, 4, Device_, Layout>*>(network[0].layer.get());
      inp_layer->set_input(tensor);
    }

    inline PtrTensorAdapter<Scalar, Device_> get_output() {
      return network.rbegin()->layer.get()->get_output();
    }

    inline void add(LayerBase<Scalar, Device_>* n, OptimizerBase<Scalar, Device_>* opt = nullptr) {
      network.push_back(NetworkNode<Scalar, Device_>(n, opt));
    }

    inline PtrTensorAdapter<Scalar, Device_> get_loss() { return loss.get_output(); }

  protected:

    // assuming that the last layer of the network is Flatten, we can get the flattened dimension
    inline int get_flat_dimension(const array<Index, Rank>& input_dims) {

      // or it will throw during the forward pass
      init();

      DeviceTensor<Scalar, Rank, Device_, Layout> inp(input_dims);
      inp.setZero();

      dynamic_cast<Input<Scalar, Rank, Device_, Layout>*>(network[0].layer.get())->set_input(inp);

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

    Network network;
    Loss loss;
    bool inited;
  };
}