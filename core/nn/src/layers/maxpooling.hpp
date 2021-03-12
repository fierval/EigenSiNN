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
  template <typename Scalar, Index Rank = 4, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class MaxPooling : public LayerBase<Scalar, Device_> {
  public:

    MaxPooling(const std::vector<long>& _extents, Index _stride, Padding2D _padding = { 0, 0 }, int _dilation = 1)
      : extents(_extents)
      , stride(_stride)
      , padding(_padding)
      , dilation(_dilation) {
      
      static_assert(Rank == 4, "MaxPooling is implemented only for Rank == 4");

    }

    void init() override {
      
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer.get_output());

      if (!params || params->check()) {
        // dimensions represented by vector for Rank = 2 or Rank = 4
        // need to prepend the right values to mimic convolutional kernel
        auto it = extents.begin();
        DSizes<Index, Rank> dims = x.dimensions();
        extents.insert(it, { dims[0], dims[1] });
        params = std::make_shared<ConvolutionParams<Rank>>(dims, vec2dims<Rank>(extents), padding, stride, dilation, false);
      }

      auto res = max_pooler.do_max_pool(x, params);

      layer_output = res.first;
      mask = res.second;
      
    }

    // for derivations
    void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(next_layer_grad);

      layer_gradient = max_pooler.do_max_pool_backward(x, mask, params);
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

    std::vector<long> extents;
    MaxPooler<Scalar, Rank, Layout, Device_> max_pooler;

    const int stride;
    const Padding2D padding;
    const int dilation;

    std::shared_ptr<ConvolutionParams<Rank>> params;
  };

}