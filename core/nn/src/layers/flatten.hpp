#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  // flatten convolution layer
  template<typename Scalar, typename Device_= DefaultDevice>
  class Flatten : public LayerBase<Device_> {

  public:
    Flatten(Dispatcher<Device_>& _device =  LayerBase::default_dispatcher) : LayerBase(_device){}

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) {
      
      TensorMap<Tensor<Scalar, 4>> orig(prev_layer_any.get_output(), vector2array<int, 4>(prev_layer_any.get_out_dims()));

      original_dimensions = orig.dimensions();

      unfolded = unfold_kernel<Scalar>(orig);

      if (are_dims_unset(prev_layer_any.get_out_dims())) {
        set_dims(prev_layer_any.get_out_dims(), array2vector<int, 2>(unfolded.dimensions()));
      }

    }

    void backward(LayerBase<Scalar, Device_>& prev_layer, LayerBase<Scalar, Device_>& next_layer_grad) {

      TensorMap<Tensor<Scalar, 2>> unf_dout(next_layer_grad.get_output(), original_dimensions);

      folded = fold_kernel<Scalar>(unf_dout, original_dimensions);
    }

    Scalar * get_output() override {
      return unfolded;
    }

    Scalar * get_loss_by_input_derivative() override {
      return folded;
    }

  private:
    Tensor<Scalar, 4> folded;
    Tensor<Scalar, 2> unfolded;
    array<Index, 4> original_dimensions;
  };


}
