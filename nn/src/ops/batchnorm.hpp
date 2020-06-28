#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include "linearops.hpp"
#include "convolutions.hpp"
#include <tuple>

namespace EigenSinn {

  typedef Eigen::Tensor<float, 1> ConvTensorSingle;

  // broadcast channel dimension - first in the list of arguments, las
  template<Index rank>
  inline ConvTensor broadcast_as_last_dim(ConvTensorSingle& t, Eigen::array<int, rank> broadcast_dims) {

    Eigen::array<Index, rank> reshaped_dims;

    reshaped_dims.fill(1);
    reshaped_dims[rank] = t.dimension(0);

    ConvTensor broadcasted = t.reshape(reshaped_dims).broadcast(broadcast_dims);
    return broadcasted;
  }

  // NHWC format
  template<typename Scalar, Index Dim>
  inline auto batch_norm(const Eigen::Tensor<Scalar, Dim>& x, float gamma, float beta, float eps, float momentum, 
    ConvTensorSingle& running_mean, ConvTensorSingle running_var, bool isTraining) {

    ConvTensorSingle mu, variance;
    ConvTensor mu_broadcasted, running_mean_broadcasted, running_var_broadcasted, x_hat, std_broadcasted;

    // get sample mean
    Eigen::array<int, Dim - 1> reduction_dims;
    Eigen::array<int, Dim> broadcast_dims;

    // we reduce by all dimensions but the last (channel)
    // and broadcast all but the last
    for (int i = 0; i < Dim - 1; i++) {
      reduction_dims[i] = i;
      broadcast_dims[i] = x.dimension(i);
    }

    broadcast_dims[Dim - 1] = 1;

    mu = x.mean(reduction_dims);
    Index batch_size = x.dimension(0);
    Index n_channels = x.dimension(3);
    Index rows = x.dimension(1);
    Index cols = x.dimension(2);

    mu_broadcasted = broadcast_as_last_dim(mu, broadcast_dims);
    variance = (x - mu_broadcasted).pow(2.).mean(reduction_dims);

    running_mean = momentum * running_mean + (1.0 - momentum) * mu;
    running_var = momentum * running_var + (1.0 - momentum) * variance;

    running_mean_broadcasted = broadcast_as_last_dim(running_mean, broadcast_dims);
    running_var_broadcasted = broadcast_as_last_dim(running_var, broadcast_dims);
    std_broadcasted = (running_var_broadcasted + eps).sqrt();


    x_hat = gamma * (x - running_mean_broadcasted) / std_broadcasted + beta;

    return std::make_tuple(x_hat, running_mean, running_var);
  }
}