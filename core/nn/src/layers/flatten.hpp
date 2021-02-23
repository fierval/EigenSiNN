#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  // flatten convolution layer
  template<typename Scalar, typename Device_= ThreadPoolDevice, int Layout = ColMajor>
  class Flatten : public LayerBase<Scalar> {

  public:
    Flatten() = default;

    void forward(LayerBase<Scalar>& prev_layer_any) {
      
      DeviceTensor<Scalar, 4, Device_, Layout> orig(prev_layer_any.get_output());

      original_dimensions = orig.dimensions();

      unfolded = unfold_kernel<Scalar>(orig);

    }

    void backward(LayerBase<Scalar>& prev_layer, std::any next_layer_grad) {

      DeviceTensor<Scalar, 2, Device_, Layout> unf_dout(next_layer_grad);

      folded = fold_kernel<Scalar>(unf_dout, original_dimensions);
    }

    std::any get_output() override {
      return unfolded;
    }

    std::any get_loss_by_input_derivative() override {
      return folded;
    }

  private:
    DeviceTensor<Scalar, 4, Device_, Layout> folded;
    DeviceTensor<Scalar, 2, Device_, Layout> unfolded;
    array<Index, 4> original_dimensions;
  };


}
