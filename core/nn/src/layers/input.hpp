#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template<typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Input : public LayerBase<Scalar, Device_> {

    public Input(array<Index, Rank> dims, Dispatcher<Device_>& _device = LayerBase<Scalar, Device_>::default_dispatcher) :
      LayerBase(_device) {

      set_dims(array2vector<int, Rank>(dims), array2vector<int, Rank>(dims));
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer_base) {};

    void backward(LayerBase<Scalar, Device_>& prev_layer, LayerBase<Scalar, Device_>& next_layer_grad) {};

    Scalar* get_output() { 
      return input.data();
    };

    void set_input(Scalar* _input) {
      
      input.resize(vector2array<int, Rank>(out_dims));

      TensorMap<Tensor<Scalar, Rank>> inp(_input, out_dims);
      input.device(dispatcher.get_device()) = inp;
    }

    Scalar* get_loss_by_input_derivative() { return nullptr; };

  private:
    Tensor<Scalar, Rank> input;

  };
}