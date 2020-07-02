#pragma once

#include "opsbase.hpp"
#include <tuple>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar>
  using TensorSingleDim = Eigen::Tensor<Scalar, 1>;

  template<Index Dim>
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
  template  <typename Scalar = float, Index Rank>
  inline Tensor<Scalar, Rank> broadcast_as_last_dim(TensorSingleDim<Scalar>& t, Eigen::array<int, Rank> broadcast_dims) {

    Eigen::array<Index, Rank> reshaped_dims;

    reshaped_dims.fill(1);
    reshaped_dims[Rank] = t.dimension(0);

    ConvTensor broadcasted = t.reshape(reshaped_dims).broadcast(broadcast_dims);
    return broadcasted;
  }

  // NHWC format
  template<typename Scalar = float, Index Rank>
  inline auto batch_norm(Tensor<Scalar, Rank>& x, TensorSingleDim<Scalar>& gamma, TensorSingleDim<Scalar>& beta, float eps, float momentum, 
    TensorSingleDim<Scalar>& running_mean, TensorSingleDim<Scalar>& running_var, bool isTraining) {

    TensorSingleDim mu, variance;
    Tensor<Scalar, Rank> mu_broadcasted, running_mean_broadcasted, running_var_broadcasted, x_hat, std_broadcasted, x_out;

    // get sample mean
    Eigen::array<int, Rank - 1> reduction_dims;
    Eigen::array<int, Rank> broadcast_dims;

    std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims(x);

    // mean
    mu = x.mean(reduction_dims);
    mu_broadcasted = broadcast_as_last_dim(mu, broadcast_dims);
    
    // variance
    variance = (x - mu_broadcasted).pow(2.).mean(reduction_dims);

    Tensor<Scalar, Rank> new_running_mean = momentum * running_mean + (1.0 - momentum) * mu;
    Tensor<Scalar, Rank> new_running_var = momentum * running_var + (1.0 - momentum) * variance;

    running_mean_broadcasted = broadcast_as_last_dim(new_running_mean, broadcast_dims);
    running_var_broadcasted = broadcast_as_last_dim(new_running_var, broadcast_dims);
    Tensor<Scalar, Rank> running_std_broadcasted = broadcast_as_last_dim((new_running_var + eps).sqrt(), broadcast_dims);


    x_hat = (x - running_mean_broadcasted) / running_std_broadcasted;
    x_out = gamma * x_hat + beta;

    return std::make_tuple(x_out, x_hat, new_running_mean, new_running_var);
  }
}