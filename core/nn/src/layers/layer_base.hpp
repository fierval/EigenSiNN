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

  template <typename Scalar>
  class LayerBase {

  public:
    virtual void init() {};

    virtual void forward(LayerBase<Scalar>& prev_layer_base) = 0;

    virtual void backward(LayerBase<Scalar>& prev_layer, std::any next_layer_grad_any) = 0;

    virtual  std::any get_output() = 0;

    virtual  std::any get_loss_by_input_derivative() = 0;

    virtual  std::any get_loss_by_weights_derivative() { return std::any(); };
    virtual  std::any get_weights() { return std::any(); };

    virtual  std::any get_loss_by_bias_derivative() { return std::any(); }
    virtual  std::any get_bias() { return std::any(); }

  protected:
    // rule of zero: https://en.cppreference.com/w/cpp/language/rule_of_three
    // since the class is polymorphic (virtual functions) avoid slicing
    // by explicity prohibiting copying
    virtual ~LayerBase() = default;
    //LayerBase(const LayerBase&) = delete;
    //LayerBase(LayerBase&&) = delete;
    //LayerBase& operator=(const LayerBase&) = delete;
    //LayerBase& operator=(LayerBase&&) = delete;


  };
}