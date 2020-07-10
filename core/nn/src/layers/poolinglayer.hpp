#pragma once

#include "layer_base.hpp"
#include <ops/poolingops.hpp>
#include <ops/conversions.hpp>

using namespace  Eigen;

namespace EigenSinn {

  // NHWC format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank>
  class MaxPoolingLayer : LayerBase {
  public:

    MaxPoolingLayer(const array<Index, Rank / 2>& _extents, Index _stride)
      : extents(_extents)
      , stride(_stride) {

    }

    void init() override {
      
    }

    void forward(std::any prev_layer) override {
      Tensor<Scalar, Rank> x = from_any<Scalar, Rank>(prev_layer);
      auto res = do_max_pool<Scalar, Rank>(x, extents, stride);

      original_dimensions = x.dimensions();
      layer_output = res.first;
      mask = res.second;
      
    }

    // for derivations
    void backward(std::any prev_layer_any, std::any next_layer_grad_any) override {
      layer_gradient = do_max_pool_backward<Scalar, Rank>(from_any<Scalar, Rank>(next_layer_grad_any), mask, original_dimensions, extents, stride);
    }

    const std::any get_output() override {
      return layer_output;
    }

  private:
    Tensor<Scalar, Rank> layer_output, layer_gradient;
    Tensor<Index, Rank> mask;

    Index stride;
    array<Index, Rank / 2> extents;
    array<Index, Rank> original_dimensions;

  };

}