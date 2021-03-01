#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  // flatten convolution layer
  template<typename Scalar, typename Device_= ThreadPoolDevice, int Layout = ColMajor>
  class Flatten : public LayerBase<Scalar, Device_> {

  public:
    Flatten() = default;

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) {
      
      DeviceTensor<Scalar, 4, Device_, Layout> orig(prev_layer_any.get_output());

      original_dimensions = orig.dimensions();

      unfolded = unfold_kernel<Scalar>(orig);

    }

    void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) {

      DeviceTensor<Scalar, 2, Device_, Layout> unf_dout(next_layer_grad);

      folded = fold_kernel<Scalar>(unf_dout, original_dimensions);
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return unfolded.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return folded.raw();
    }

  private:
    DeviceTensor<Scalar, 4, Device_, Layout> folded;
    DeviceTensor<Scalar, 2, Device_, Layout> unfolded;
    array<Index, 4> original_dimensions;
  };


}
