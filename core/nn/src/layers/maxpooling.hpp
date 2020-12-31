#pragma once

#include "layer_base.hpp"
#include <ops/maxpoolingops.hpp>
#include <ops/conversions.hpp>

using namespace  Eigen;

namespace EigenSinn {

  // NHWC format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank, int Layout = ColMajor, typename Device_ = DefaultDevice>
  class MaxPooling : public LayerBase<Scalar> {
  public:

    MaxPooling(const array<Index, Rank / 2>& _extents, Index _stride)
      : extents(_extents)
      , stride(_stride)
      , original_dimensions({0}) {

    }

    void init() override {
      
    }

    void forward(LayerBase<Scalar>& prev_layer) override {

      DeviceTensor<Device_, Scalar, Rank, Layout> x(prev_layer.get_output());
      auto res = max_pooler.do_max_pool(x, extents, stride);

      original_dimensions = x.dimensions();
      layer_output = res.first;
      mask = res.second;
      
    }

    // for derivations
    void backward(LayerBase<Scalar>& prev_layer, std::any next_layer_grad) override {

      DeviceTensor<Device_, Scalar, Rank, Layout> x(next_layer_grad);

      layer_gradient = max_pooler.do_max_pool_backward(x, mask, original_dimensions, extents, stride);
    }

    std::any get_output() override {
      return layer_output;
    }

    std::any get_loss_by_input_derivative() {
      return layer_gradient;
    }


  private:
    DeviceTensor<Device_, Scalar, Rank, Layout> layer_output, layer_gradient;
    DeviceTensor<Device_, Index, Rank, Layout> mask;

    Index stride;
    array<Index, Rank / 2> extents;
    array<Index, Rank> original_dimensions;
    MaxPooler<Scalar, Rank, Layout, Device_> max_pooler;
  };

}