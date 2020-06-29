#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include "linearops.hpp"
#include "convolutions.hpp"
#include <tuple>

namespace EigenSinn {

  typedef Eigen::Tensor<float, 1> TensorSingleDim;

  template <Index Rank>
  using BnTensor = Tensor<float, Rank>;

  template<Index Dim = 2>
  inline auto get_broadcast_and_reduction_dims(Eigen::Tensor<float, Dim> x) {
    // we reduce by all dimensions but the last (channel)
    // and broadcast all but the last
    
    for (int i = 0; i < Dim - 1; i++) {
      reduction_dims[i] = i;
      broadcast_dims[i] = x.dimension(i);
    }

    broadcast_dims[Dim - 1] = 1;
    return std::make_tuple(reduction_dims, broadcast_dims)
  }

  // broadcast channel dimension - first in the list of arguments, las
  template<Index Rank>
  inline BnTensor<Rank> broadcast_as_last_dim(TensorSingleDim& t, Eigen::array<int, Rank> broadcast_dims) {

    Eigen::array<Index, Rank> reshaped_dims;

    reshaped_dims.fill(1);
    reshaped_dims[Rank] = t.dimension(0);

    ConvTensor broadcasted = t.reshape(reshaped_dims).broadcast(broadcast_dims);
    return broadcasted;
  }

  // NHWC format
  template<Index Rank>
  inline auto batch_norm(BnTensor<Rank>& x, TensorSingleDim& gamma, TensorSingleDim& beta, float eps, float momentum, 
    TensorSingleDim& running_mean, TensorSingleDim running_var, bool isTraining) {

    TensorSingleDim mu, variance;
    Tensor<Scalar, Dim> mu_broadcasted, running_mean_broadcasted, running_var_broadcasted, x_hat, std_broadcasted, x_out;

    // get sample mean
    Eigen::array<int, Dim - 1> reduction_dims;
    Eigen::array<int, Dim> broadcast_dims;

    std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims(x);

    // mean
    mu = x.mean(reduction_dims);
    Index batch_size = x.dimension(0);
    Index n_channels = x.dimension(3);
    Index rows = x.dimension(1);
    Index cols = x.dimension(2);

    mu_broadcasted = broadcast_as_last_dim(mu, broadcast_dims);
    
    // variance
    variance = (x - mu_broadcasted).pow(2.).mean(reduction_dims);

    running_mean = momentum * running_mean + (1.0 - momentum) * mu;
    running_var = momentum * running_var + (1.0 - momentum) * variance;

    running_mean_broadcasted = broadcast_as_last_dim(running_mean, broadcast_dims);
    running_var_broadcasted = broadcast_as_last_dim(running_var, broadcast_dims);
    running_std_broadcasted = (running_var_broadcasted + eps).sqrt();


    x_hat = (x - running_mean_broadcasted) / running_std_broadcasted;
    x_out = gamma * x_hat + beta;

    return std::make_tuple(x_out, x_hat, running_mean, running_var);
  }
}