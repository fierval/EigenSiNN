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
  template <typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class MaxPooling : public LayerBase<Scalar, Device_> {
  public:

    MaxPooling(const array<Index, Rank / 2>& _extents, Index _stride, 
      Dispatcher<Device_>& _device =  LayerBase::default_dispatcher)
      : extents(_extents)
      , LayerBase(_device)
      , stride(_stride)
      , original_dimensions({0})
    , max_pooler() {

    }

    void init() override {
      
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      if (are_dims_unset(prev_layer.get_out_dims())) {
        set_in_dims(prev_layer.get_out_dims());
      }

      TensorMap<Tensor<Scalar, Rank>> x(prev_layer.get_output(), vector2array<Rank>(in_dims));
      auto res = max_pooler.do_max_pool(x, extents, stride, dispatcher.get_device());

      original_dimensions = x.dimensions();
      layer_output = res.first;
      mask = res.second;
      
      set_out_dims(array2vector<Rank>(mask.dimensions()));
    }

    // for derivations
    void backward(LayerBase<Scalar, Device_>& prev_layer, Scalar * next_layer_grad) override {

      TensorMap<Tensor<Scalar, Rank>> x(next_layer_grad, vector2array<Rank>(out_dims));

      layer_gradient = max_pooler.do_max_pool_backward(x, mask, original_dimensions, extents, stride);
    }

    Scalar * get_output() override {
      return layer_output.data();
    }

    Scalar * get_loss_by_input_derivative() {
      return layer_gradient.data();
    }


  private:
    Tensor<Scalar, Rank> layer_output, layer_gradient;
    Tensor<Index, Rank> mask;

    Index stride;
    array<Index, Rank / 2> extents;
    array<Index, Rank> original_dimensions;
    MaxPooler<Scalar, Rank, Device_> max_pooler;
  };

}