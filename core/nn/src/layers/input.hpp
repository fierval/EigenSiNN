#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>

using std::unique_ptr;

namespace EigenSinn {

  template<typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Input : public LayerBase<Scalar, Device_> {

  public:
    
    Input(Dispatcher<Device_>& _device = LayerBase<Scalar, Device_>::default_dispatcher) 
      : LayerBase<Scalar, Device_>(_device) {
      
      input = nullptr;
    }

    Scalar* get_output() { 
      return input->data();
    };

    Scalar* get_loss_by_input_derivative() { return nullptr; };

    /// <summary>
    /// Grab data from the existing tensor
    /// </summary>
    /// <param name="inp_tensor"></param>
    void set_input(Tensor<Scalar, Rank>& inp_tensor) {
      
      set_input(inp_tensor.data(), inp_tensor.dimensions());
    }

    // Required overrides
    void forward(LayerBase<Scalar, Device_>& prev_layer_base) {};
    void backward(LayerBase<Scalar, Device_>& prev_layer, Scalar* next_layer_grad) {};

  private:

    void set_input(Scalar* _input, array<Index, Rank>& _out_dims) {

      set_dims(_out_dims, _out_dims);
      set_input(input);

      input = std::make_unique<TensorView<Scalar, Rank>>(_input, vector2array<Rank>(out_dims));

    }

    unique_ptr<TensorMap<Tensor<Scalar, Rank>>> input;


  };
}