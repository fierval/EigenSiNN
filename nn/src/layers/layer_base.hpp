#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

namespace EigenSinn {

  template<int Rank>
  class LayerBase {

  public:

    virtual void forward(Tensor<float, Rank>& prev_layer) = 0;

    virtual void backward(const Tensor<float, Rank>& prev_layer, const Tensor<float, Rank>& next_layer_grad) = 0;
  };
}