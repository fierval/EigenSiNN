#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <any>

using namespace Eigen;

namespace EigenSinn {

  class LayerBase {

  public:
    virtual void init() {};

    virtual void forward(std::any prev_layer_any) = 0;

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


  };
}