#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <ops/opsbase.hpp>
#include <ops/threadingdevice.hpp>

using namespace Eigen;

namespace EigenSinn {

  template <typename Device_ = Dispatcher<DefaultDevice>>
  class LayerBase {

  public:
    virtual void init() {};

    virtual void forward(std::any prev_layer_any) = 0;

    virtual void forward(void* prev_layer) {};

    virtual void backward(std::any prev_layer, std::any next_layer_grad) = 0;

    virtual  std::any get_weights() { return std::any(); }

    virtual  std::any get_bias() { return std::any(); }

    virtual void set_weights(const std::any _weights) {}

    virtual void set_bias(const std::any _bias) {}

    virtual  std::any get_output() = 0;

    virtual  std::any get_loss_by_input_derivative() = 0;

    virtual  std::any get_loss_by_weights_derivative() { return std::any(); };

    virtual  std::any get_loss_by_bias_derivative() { return std::any(); }

    virtual  bool has_bias() { return false; }

    inline static Dispatcher<DefaultDevice> default_dispatcher = Dispatcher<DefaultDevice>();

    const std::vector& get_in_dims() { return in_dims; }
    const std::vector& get_out_dims() { return out_dims; }

    void set_in_dims(std::vector<int>& _in_dims) { in_dims = _in_dims; }
    void set_out_dims(std::vector<int>& _out_dims) { out_dims = _out_dims; }

  protected:
    Dispatcher<Device_>& dispatcher;
    std::vector<int> in_dims, out_dims;

    LayerBase(Dispatcher<Device_>& _dispatcher) : dispatcher(_dispatcher) {
    }

  };

}