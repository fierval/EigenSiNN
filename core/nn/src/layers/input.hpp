#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template<typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Input : public LayerBase<Scalar, Device_> {

  public:
    
    Input(array<Index, Rank> dims, Dispatcher<Device_>& _device = LayerBase<Scalar, Device_>::default_dispatcher) 
      : Input(_device) {

      std::vector<Index> v_dims = array2vector(dims);
      set_dims(v_dims, v_dims);
    }

    Input(Dispatcher<Device_>& _device = LayerBase<Scalar, Device_>::default_dispatcher) 
      : LayerBase<Scalar, Device_>(_device) {
      
      input = nullptr;
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer_base) {};

    void backward(LayerBase<Scalar, Device_>& prev_layer, Scalar * next_layer_grad) {};

    Scalar* get_output() { 
      return input->data();
    };

    void set_input(Scalar* _input) {
      
      if (input != nullptr) {
        delete input;
      }
      input = new TensorMap<Tensor<Scalar, Rank>>(_input, vector2array<Rank>(out_dims));
    }

    Scalar* get_loss_by_input_derivative() { return nullptr; };

    void set_input(Scalar* _input, array<Index, Rank>& _out_dims) {
      
      set_dims(_out_dims, _out_dims);
      set_input(input);
    }

    ~Input() {
      if (input != nullptr) {
        delete input;
        input = nullptr;
      }
    };

  private:
    TensorMap<Tensor<Scalar, Rank>> * input;

  };
}