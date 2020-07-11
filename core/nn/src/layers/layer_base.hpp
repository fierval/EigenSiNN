#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <any>

using namespace Eigen;

namespace EigenSinn {

  class LayerBase {

  public:
    virtual void init() = 0;

    virtual void forward(std::any prev_layer) = 0;

    virtual void backward(std::any prev_layer, std::any next_layer_grad) = 0;

    virtual const std::any get_output() = 0;

    virtual const std::any get_loss_by_input_derivative() = 0;

  };
}