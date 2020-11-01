#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>

using std::unique_ptr;

namespace EigenSinn {

  template<typename Scalar, Index Rank, typename Device_ = DefaultDevice>
  class Input : public LayerBase<Scalar, Device_> {

  public:
    
    Input(Dispatcher<Device_>& _device = LayerBase<Scalar, Device_>::default_dispatcher) 
      : LayerBase<Scalar, Device_>(_device)
      , has_data_ownership(false) {
      
    }

    Scalar* get_output() { 
      return input->data();
    };

    Scalar* get_loss_by_input_derivative() { return nullptr; };

    /// <summary>
    /// Grab data from the existing tensor
    /// </summary>
    /// <param name="inp_tensor">input data tensor</param>
    /// <param name= "move_to_device">whether to move the original memory to device before setting</param>
    void set_input(Tensor<Scalar, Rank>& inp_tensor, bool move_to_device=false) {
      
      Scalar* data;

      if (move_to_device) {
        // free whatever was there before
        if (input && has_data_ownership) {
          device.deallocate(input.data());
          input.reset(nullptr);
        }
        size_t alloc_size = inp_tensor.dimensions().TotalSize() * sizeof(Scalar);
        data = static_cast<Scalar*>(device.allocate(alloc_size));
        device.memcpyHostToDevice(data, inp_tensor.data(), alloc_size);
      }
      else {
        data = inp_tensor.data();
      }

      has_data_ownership = move_to_device;
      set_input(data, inp_tensor.dimensions());
    }

    // Required overrides
    void forward(LayerBase<Scalar, Device_>& prev_layer_base) {};
    void backward(LayerBase<Scalar, Device_>& prev_layer, Scalar* next_layer_grad) {};

    virtual ~Input() {
      if (input) {
        device.deallocate(input.data());
      }
    }

  private:

    void set_input(Scalar* _input, array<Index, Rank>& _out_dims) {

      set_dims(_out_dims, _out_dims);
      input.reset(new TensorView<Scalar, Rank>(_input, vector2array<Rank>(out_dims)));
    }

    unique_ptr<TensorView<Scalar, Rank>> input;
    bool has_data_ownership;

  };
}