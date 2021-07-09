#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>
#include <onnx/model.h>

using std::unique_ptr;

namespace EigenSinn {

  template<typename Scalar, typename Device_ = ThreadPoolDevice>
  class Input : public LayerBase<Scalar, Device_> {

  public:

    Input() : LayerBase<Scalar, Device_>("Input") {};

    PtrTensorAdapter<Scalar, Device_> get_output() {
      return input;
    };

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() { return nullptr; };

    /// <summary>
    /// Grab data from the existing tensor
    /// </summary>
    /// <param name="inp_tensor">input data tensor</param>
    /// <param name= "move_to_device">whether to move the original memory to device before setting</param>
    template<int Rank, int Layout = RowMajor>
    void set_input(Tensor<Scalar, Rank, Layout>& inp_tensor) {

      DeviceTensor<Scalar, Rank, Device_, Layout> temp(inp_tensor);
      input = temp.raw();
    }

    template<int Rank, int Layout = RowMajor>
    void set_input(const DeviceTensor<Scalar, Rank, Device_, Layout>& inp_tensor) {

      input = const_cast<DeviceTensor<Scalar, Rank, Device_, Layout>&>(inp_tensor).raw();
    }

    // Required overrides
    void forward(PtrTensorAdapter<Scalar, Device_>& prev_layer_base) override {};
    void backward(PtrTensorAdapter<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) override {};

    std::vector<Index> get_dims() {
      return input->get_dims();
    }

  private:

    PtrTensorAdapter<Scalar, Device_> input;
  };
}