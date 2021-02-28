#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <device/device_tensor.hpp>

using namespace Eigen;

namespace EigenSinn {

  enum class DebugInit : int {
    False = 0,
    Weights = 1,
    Bias = 2
  };

  template <typename Scalar, typename Device_ = ThreadPoolDevice>
  class LayerBase {

  public:
    virtual void init() {};

    virtual void forward(LayerBase<Scalar, Device_>& prev_layer_base) = 0;

    virtual void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) = 0;

    virtual  PtrTensorAdapter<Scalar, Device_> get_output() = 0;

    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() = 0;

    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_weights_derivative() { return PtrTensorAdapter<Scalar, Device_>(); };
    virtual  PtrTensorAdapter<Scalar, Device_> get_weights() { return PtrTensorAdapter<Scalar, Device_>(); };

    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_bias_derivative() { return PtrTensorAdapter<Scalar, Device_>(); }
    virtual  PtrTensorAdapter<Scalar, Device_> get_bias() { return PtrTensorAdapter<Scalar, Device_>(); }

    virtual void set_weights(PtrTensorAdapter<Scalar, Device_>&) {}
    virtual void set_bias(PtrTensorAdapter<Scalar, Device_>&) {}

    virtual ~LayerBase() = default;

  protected:
    // rule of zero: https://en.cppreference.com/w/cpp/language/rule_of_three
    // since the class is polymorphic (virtual functions) avoid slicing
    // by explicity prohibiting copying
    //LayerBase(const LayerBase&) = delete;
    //LayerBase(LayerBase&&) = delete;
    //LayerBase& operator=(const LayerBase&) = delete;
    //LayerBase& operator=(LayerBase&&) = delete;


  };
}