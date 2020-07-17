#pragma once

#include "opsbase.hpp"
#include <ops/conversions.hpp>

namespace EigenSinn {

  template <typename Scalar>
  using TensorSingleDim = Eigen::Tensor<Scalar, 1>;

  template<Index Dim>
  inline auto get_broadcast_and_reduction_dims(Eigen::Tensor<float, Dim> x) {
    // we reduce by all dimensions but the last (channel)
    // and broadcast all but the last
    
    Eigen::array<int, Dim - 1> reduction_dims;
    Eigen::array<int, Dim> broadcast_dims;

    for (int i = 0; i < Dim - 1; i++) {
      // channel-first tensor: NCHW, need to skip channel dimension
      reduction_dims[i] = i == 0 ? i : i + 1;
      broadcast_dims[i] = x.dimension(i);
    }

    // channel-first format: NCHW
    broadcast_dims[(int)ImageDims::channel] = 1;
    return std::make_tuple(reduction_dims, broadcast_dims);
  }

  // broadcast channel dimension - first in the list of arguments, las
  template  <typename Scalar = float, Index Rank>
  inline Tensor<Scalar, Rank> broadcast_as_last_dim(TensorSingleDim<Scalar>& t, Eigen::array<int, Rank> broadcast_dims) {

    Eigen::array<Index, Rank> reshaped_dims;

    reshaped_dims.fill(1);
    reshaped_dims[(int)ImageDims::channel] = t.dimension(0);

    Tensor<Scalar, Rank> broadcasted = t.reshape(reshaped_dims).broadcast(broadcast_dims);
    return broadcasted;
  }

  // NHWC format
  template<typename Scalar = float, Index Rank>
  inline auto batch_norm(Tensor<Scalar, Rank>& x, TensorSingleDim<Scalar>& gamma, TensorSingleDim<Scalar>& beta, float eps, float momentum, 
    TensorSingleDim<Scalar>& running_mean, TensorSingleDim<Scalar>& running_var, bool is_training) {

    TensorSingleDim<Scalar> mu, variance, std;
    Tensor<Scalar, Rank> mu_broadcasted, mean_broadcasted, x_hat, std_broadcasted, x_out;

    TensorSingleDim<Scalar> new_running_mean = running_mean;
    TensorSingleDim<Scalar> new_running_var = running_var;;

    // get sample mean
    Eigen::array<int, Rank - 1> reduction_dims;
    Eigen::array<int, Rank> broadcast_dims;

    std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims(x);

    // if training we compute all the values.
    // otherwise use their running analogs
    if (is_training) {
      // mean
      mu = x.mean(reduction_dims);
      mu_broadcasted = broadcast_as_last_dim(mu, broadcast_dims);

      // variance
      variance = (x - mu_broadcasted).pow(2.).mean(reduction_dims);

      new_running_mean = momentum * running_mean + (1.0 - momentum) * mu;
      new_running_var = momentum * running_var + (1.0 - momentum) * variance;
      
      std = (variance + eps).sqrt();
      mean_broadcasted = broadcast_as_last_dim<Scalar, Rank>(mu, broadcast_dims);
    }
    else {
      std = (running_var + eps).sqrt();
      mean_broadcasted = broadcast_as_last_dim<Scalar, Rank>(running_mean, broadcast_dims);
    }
    std_broadcasted = broadcast_as_last_dim<Scalar, Rank>(std, broadcast_dims);

    
    Tensor<Scalar, Rank> gamma_broadcasted = broadcast_as_last_dim<Scalar, Rank>(gamma, broadcast_dims);
    Tensor<Scalar, Rank> beta_broadcasted = broadcast_as_last_dim<Scalar, Rank>(beta, broadcast_dims);

    x_hat = (x - mean_broadcasted) / std_broadcasted;
    x_out = gamma_broadcasted * x_hat + beta_broadcasted;

    return std::make_tuple(x_out, x_hat, new_running_mean, new_running_var, mu, variance);
  }
}