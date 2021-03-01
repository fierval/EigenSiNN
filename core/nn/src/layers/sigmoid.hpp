#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class Sigmoid : public LayerBase<Scalar, Device_> {
  public:
    // leaky relu if necessary
    Sigmoid() :
      inited(false)
    {}

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> prev_layer(prev_layer_any.get_output());

      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

      layer_output.view() = prev_layer->sigmoid();
    }


    void backward(LayerBase<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> next_layer_grad(next_layer_grad_any);

      layer_grad.view() = *next_layer_grad * *layer_output * (*ones - *layer_output);
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output;
    };

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return layer_grad;
    };


  private:
    void init_cached(const DeviceTensor<Scalar, Rank, Device_, Layout>& prev_layer)
    {
      ones.resize(prev_layer.dimensions());
      ones.setConstant(1);

      layer_output.resize(prev_layer.dimensions());
      layer_grad.resize(prev_layer.dimensions());
    }

    bool inited;
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_grad;
    DeviceTensor<Scalar, Rank, Device_, Layout> ones;
  };


}