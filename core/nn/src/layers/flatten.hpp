#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  // flatten convolution layer
  template<typename Scalar, int Layout = ColMajor, typename Device_= DefaultDevice>
  class Flatten : public LayerBase<Scalar> {

  public:
    Flatten() = default;

    void forward(LayerBase<Scalar>& prev_layer_any) {
      
      DeviceTensor<Device_, Scalar, 4, Layout> orig(prev_layer_any.get_output());

      original_dimensions = orig.dimensions();

      unfolded = unfold_kernel<Scalar>(orig);

    }

    void backward(LayerBase<Scalar>& prev_layer, std::any next_layer_grad) {

      DeviceTensor<Device_, Scalar, 2, Layout> unf_dout(next_layer_grad);

      folded = fold_kernel<Scalar>(unf_dout, original_dimensions);
    }

    std::any get_output() override {
      return unfolded;
    }

    std::any get_loss_by_input_derivative() override {
      return folded;
    }

  private:
    DeviceTensor<Device_, Scalar, 4, Layout> folded;
    DeviceTensor<Device_, Scalar, 2, Layout> unfolded;
    array<Index, 4> original_dimensions;
  };


}
