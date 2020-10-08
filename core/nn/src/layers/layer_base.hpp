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

    virtual void backward(LayerBase<Scalar, Device_>& prev_layer, LayerBase<Scalar, Device_>& next_layer_grad) { 
      return backward(prev_layer, next_layer_grad.get_output()); }

    virtual void backward(LayerBase<Scalar, Device_>& prev_layer, Scalar * next_layer_grad) = 0;

    virtual  Scalar* get_weights() { return nullptr; }

    virtual  Scalar* get_bias() { return nullptr; }

    virtual  Scalar* get_output() = 0;

    virtual  Scalar* get_loss_by_input_derivative() = 0;

    virtual  Scalar* get_loss_by_weights_derivative() { return nullptr; };

    virtual  Scalar* get_loss_by_bias_derivative() { return nullptr; }

    inline static Dispatcher<DefaultDevice> default_dispatcher = Dispatcher<DefaultDevice>();

    std::vector<Index>& get_in_dims() { return in_dims; }
    std::vector<Index>& get_out_dims() { return out_dims; }

    std::vector<Index>& get_bias_dims() { return bias_dims; }
    std::vector<Index>& get_weight_dims() { return weight_dims; }

    void set_in_dims(std::vector<Index>& _in_dims) { in_dims = _in_dims; }
    void set_out_dims(std::vector<Index>& _out_dims) { out_dims = _out_dims; }

    void set_dims(std::vector<Index>& _in_dims, std::vector<Index>& _out_dims) {
      set_in_dims(_in_dims);
      set_out_dims(_out_dims);
    }

    void set_dims(LayerBase<Scalar, Device_>& layer) {
      if (are_dims_unset(layer.get_out_dims())) {
        set_dims(layer.get_out_dims(), layer.get_out_dims());
      }
    }

    void set_bias_dims(std::vector<Index>& _bias_dims) { bias_dims = _bias_dims; }
    void set_weight_dims(std::vector<Index>& _weight_dims) { weight_dims = _weight_dims; }

  protected:
    Dispatcher<Device_>& dispatcher;
    std::vector<Index> in_dims, out_dims, bias_dims, weight_dims;

    bool are_dims_unset(std::vector<Index>& dims) { return in_dims.size() == 0 || dims[0] != in_dims[0]; }

    LayerBase(Dispatcher<Device_>& _dispatcher) : dispatcher(_dispatcher) {
    }

  };

}