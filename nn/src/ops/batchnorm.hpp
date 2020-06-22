#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include "linearops.hpp"
#include "convolutions.hpp"
#include <tuple>

namespace EigenSinn {

  typedef Eigen::Tensor<float, 1> ConvTensorSingle;

  inline void batch_norm(const ConvTensor& x, float gamma, float beta, float eps, float momentum, 
    ConvTensorSingle& running_mean, ConvTensorSingle running_var, bool isTraining) {

    ConvTensorSingle mu, std, variance;

    // get sample mean
    Eigen::array<int, 3> reduction_dims({ 0, 1, 2 });
    mu = x.mean(reduction_dims);
    
    Index batch_size = x.dimension(0);
    Index n_channels = x.dimension(3);
    std.resize(batch_size);
    variance.resize(batch_size);

    // get sample variance
    // unfortunately no var reducer that we need so we have to do it manually
    for (int i = 0; i < n_channels; i++) {
      
      Eigen::Tensor<float, 3> conv_layer(x.dimension(1), x.dimension(2), batch_size);
      for (int j = 0; j < batch_size; j++) {

        Eigen::array<Index, 4> offsets = {j, 0, 0, i};
        Eigen::array<Index, 4> extents = { 1, x.dimension(1), x.dimension(2), 1 };
        conv_layer.chip(j, 2) = x.slice(offsets, extents);
      }

      variance.chip(i, 0) = (conv_layer - mu(i)).pow(2.).mean();
    }

    std = (variance + eps).sqrt();
    running_mean = momentum * running_mean + (1.0 - momentum) * mu;
    running_var = momentum * running_var + (1.0 - momentum) * variance;

    // TODO: broadcast things.
  }
}