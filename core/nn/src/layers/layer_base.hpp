#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <ops/threadingdevice.hpp>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, typename Device_ = Dispatcher<DefaultDevice>>
  class LayerBase {

  public:
    virtual void init() {};

    virtual void forward(LayerBase<Scalar, Device_>& prev_layer_base) = 0;

    virtual void backward(LayerBase<Scalar, Device_>& prev_layer, LayerBase<Scalar, Device_>& next_layer_grad) = 0;

    virtual  Scalar* get_weights() { return nullptr; }

    virtual  Scalar* get_bias() { return nullptr; }

    virtual void set_weights(const Scalar* _weights) {}

    virtual void set_bias(const Scalar* _bias) {}

    virtual  Scalar* get_output() = 0;

    virtual  Scalar* get_loss_by_input_derivative() = 0;

    virtual  Scalar* get_loss_by_weights_derivative() { return nullptr; };

    virtual  Scalar* get_loss_by_bias_derivative() { return nullptr; }

    inline static Dispatcher<DefaultDevice> default_dispatcher = Dispatcher<DefaultDevice>();

    const std::vector<int>& get_in_dims() { return in_dims; }
    const std::vector<int>& get_out_dims() { return out_dims; }

    void set_in_dims(const std::vector<int>& _in_dims) { in_dims = _in_dims; }
    void set_out_dims(const std::vector<int>& _out_dims) { out_dims = _out_dims; }
    
    void set_dims(const std::vector<int>& _in_dims, const std::vector<int>& _out_dims) {
      set_in_dims(_in_dims);
      set_out_dims(_out_dims);
    }

  protected:
    Dispatcher<Device_>& dispatcher;
    std::vector<int> in_dims, out_dims;

    bool are_dims_unset(const vector<int>& dims) { return in_dims.size() == 0 || dims[0] != in_dims[0]; }
  
    LayerBase(Dispatcher<Device_>& _dispatcher) : dispatcher(_dispatcher) {
    }

  };

}