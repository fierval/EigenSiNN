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

    virtual const std::any get_weights() { return std::any(); }
    
    virtual const std::any get_bias() { return std::any(); }

    virtual const std::any get_output() = 0;

    virtual const std::any get_loss_by_input_derivative() = 0;

    virtual const std::any get_loss_by_weights_derivative() { return std::any(); };

    virtual const std::any get_loss_by_bias_derivative() { return std::any(); }

    virtual const bool has_bias() { return false; }

  };
}