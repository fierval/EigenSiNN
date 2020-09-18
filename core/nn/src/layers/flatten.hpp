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

    void forward(std::any prev_layer_any) {
      
      Tensor<Scalar, 4> orig = from_any<Scalar, 4>(prev_layer_any);
      original_dimensions = orig.dimensions();

      unfolded = unfold_kernel<Scalar>(orig);
    }

    void backward(std::any prev_layer, std::any next_layer_grad) {
      Tensor<Scalar, 2> unf_dout = from_any<Scalar, 2>(next_layer_grad);

      folded = fold_kernel<Scalar>(unf_dout, original_dimensions);
    }

    virtual std::any get_output() {
      return unfolded;
    }

    virtual std::any get_loss_by_input_derivative() {
      return folded;
    }



  private:
    Tensor<Scalar, 4> folded;
    Tensor<Scalar, 2> unfolded;
    array<Index, 4> original_dimensions;
  };


}
