#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"

namespace EigenSinn {
  typedef Eigen::Tensor<float, 1> ConvBias;
  typedef
  class ConvolutionalLayer : LayerBase {

  public:
    ConvolutionalLayer(Index _batch_size, Dim2D _wh, Index _in_channels, Index _out_channels, bool _use_bias = false) 
    : bias(_batch_size),
      kernel(_out_channels, _wh.first, _wh.second, _in_channels)
    {

    }

  private:
    ConvTensor kernel;
    ConvBias bias;
  };
}