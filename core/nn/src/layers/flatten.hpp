#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  // flatten convolution layer
  template<typename Scalar>
  class Flatten : public LayerBase {

    void forward(std::any prev_layer_any) {
      
      Tensor<Scalar, 4> orig = from_any<Scalar, 4>(prev_layer_any);
      original_dimensions = orig.dimension();

      unfolded = unf_conv_res(orig);
    }

    void backward(std::any prev_layer, std::any next_layer_grad) {
      Tensor<Scalar, 2> unf_dout = from_any<Scalar, 2>(next_layer_grad);

      folded = fold_conv_res(unf_dout, original_dimensions);
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
