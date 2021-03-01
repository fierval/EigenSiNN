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
  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class MaxPooling : public LayerBase<Scalar, Device_> {
  public:

    MaxPooling(const array<Index, Rank / 2>& _extents, Index _stride)
      : extents(_extents)
      , stride(_stride)
      , original_dimensions({0}) {

    }

    void init() override {
      
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer.get_output());
      auto res = max_pooler.do_max_pool(x, extents, stride);

      original_dimensions = x.dimensions();
      layer_output = res.first;
      mask = res.second;
      
    }

    // for derivations
    void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(next_layer_grad);

      layer_gradient = max_pooler.do_max_pool_backward(x, mask, original_dimensions, extents, stride);
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() {
      return layer_gradient.raw();
    }


  private:
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_gradient;
    DeviceTensor<Index, Rank, Device_, Layout> mask;

    Index stride;
    array<Index, Rank / 2> extents;
    array<Index, Rank> original_dimensions;
    MaxPooler<Scalar, Rank, Layout, Device_> max_pooler;
  };

}