#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>

using std::unique_ptr;

namespace EigenSinn {

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class Input : public LayerBase<Scalar> {

  public:

    Input() = default;

    std::any get_output() {
      return input;
    };

    std::any get_loss_by_input_derivative() { return std::any(); };

    /// <summary>
    /// Grab data from the existing tensor
    /// </summary>
    /// <param name="inp_tensor">input data tensor</param>
    /// <param name= "move_to_device">whether to move the original memory to device before setting</param>
    void set_input(const Tensor<Scalar, Rank, Layout>& inp_tensor) {

      input.set_from_host(inp_tensor);
    }

    void set_input(const DeviceTensor<Scalar, Rank, Device_, Layout>& inp_tensor) {

      input = inp_tensor;
    }

    // Required overrides
    void forward(LayerBase<Scalar>& prev_layer_base) override {};
    void backward(LayerBase<Scalar>& prev_layer, std::any next_layer_grad) override {};

  private:

    DeviceTensor<Scalar, Rank, Device_, Layout> input;
  };
}