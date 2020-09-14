#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>
#include <cstdlib>
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>

using namespace  Eigen;

namespace EigenSinn {

  // NHWC format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Dropout : public LayerBase<Device_> {
  public:

    Dropout(const Device_& _device = DefaultDevice())
      : LayerBase(_device)
      , is_training(false)
      , inited(false) {
    }

    void init(const Tensor<Scalar, Rank>& x)  {

      using std::begin;
      using std::end;

      layer_gradient.resize(x.dimensions());
      layer_output.resize(x.dimensions());
      mask.resize(x.dimensions());

      flat_dim = std::accumulate(begin(mask.dimensions()), end(mask.dimensions()), 1, std::multiplies<Index>());
      range.resize(flat_dim);
      std::iota(begin(range), end(range), 0);
    }

    void forward(std::any prev_layer) override {
      Tensor<Scalar, Rank> x = from_any<Scalar, Rank>(prev_layer);
      
      if (!inited || x.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init(x);
      }
      
      if (!is_training) { return; }

      TensorMap<Tensor<byte, 1>> flat_mask(mask.data(), flat_dim);

      std::for_each(std::execution::par, begin(range), end(range), [&](Index i) {
        flat_mask[i] = std::rand() & 0x1;
        });

      layer_output.device(device) = mask * x;
    }

    // for derivations
    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

      Tensor<Scalar, Rank> next_layer_grad = from_any<Scalar, Rank>(next_layer_grad_any);

      if (!is_training) { return; }

      layer_gradient.device(device) = mask * next_layer_grad;
    }

    std::any get_output() override {
      return layer_output;
    }

    std::any get_loss_by_input_derivative() {
      return layer_gradient;
    }

    void set_training(bool _is_training) { 
      is_training = _is_training;
    }

    const bool get_training() {
      return is_training;
    }

  private:
    Tensor<byte, Rank> mask;
    Tensor<Scalar, Rank> layer_output, layer_gradient;
    bool is_training, inited;
    Index flat_dim;
    std::vector<Index> range;
  };

}