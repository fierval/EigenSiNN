#pragma once

#include "layer_base.hpp"
#include <ops/poolingops.hpp>

using namespace  Eigen;

namespace EigenSinn {

  // NHWC format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <Index Rank>
  class MaxPoolingLayer : LayerBase {
  public:

    MaxPoolingLayer(const array<Index, Rank / 2>& _extents, Index _stride)
      : extents(_extents)
      , stride(_stride) {

    }

    void init() override {
    }

    void forward(std::any prev_layer) override {
      layer_output = do_pool(prev_layer, extents, stride);
    }

    // for derivations
    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {

    }

    template <Index Rank = 2>
    NnTensor<Rank>& get_output() {
      return layer_output;
    }

  private:
    NnTensor<Rank> layer_output, layer_gradient, xhat;

    const Index stride;
    const array<Index, Rank / 2> extents;

  };

}