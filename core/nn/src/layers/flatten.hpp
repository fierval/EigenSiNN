#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  // flatten convolution layer
  template<typename Scalar, typename Device_= DefaultDevice>
  class Flatten : public LayerBase<Scalar, Device_> {

  public:
    Flatten(Dispatcher<Device_>& _device =  LayerBase::default_dispatcher) : LayerBase(_device){}

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) {
      
      TensorMap<Tensor<Scalar, 4>> orig(prev_layer_any.get_output(), vector2array< 4>(prev_layer_any.get_out_dims()));

      original_dimensions = orig.dimensions();

      unfolded = unfold_kernel<Scalar>(orig);

      if (are_dims_unset(prev_layer_any.get_out_dims())) {
        set_dims(prev_layer_any.get_out_dims(), array2vector<2>(unfolded.dimensions()));
      }

    }

    void backward(LayerBase<Scalar, Device_>& prev_layer, Scalar * next_layer_grad) {

      TensorMap<Tensor<Scalar, 2>> unf_dout(next_layer_grad, vector2array<2>(out_dims));

      folded = fold_kernel<Scalar>(unf_dout, original_dimensions);
    }

    Scalar * get_output() override {
      return unfolded.data();
    }

    Scalar * get_loss_by_input_derivative() override {
      return folded.data();
    }

  private:
    Tensor<Scalar, 4> folded;
    Tensor<Scalar, 2> unfolded;
    array<Index, 4> original_dimensions;
  };


}
