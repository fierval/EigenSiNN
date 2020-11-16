#pragma once

#include "opsbase.hpp"
#include <device/tensor_view.hpp>
#include "conversions.hpp"

using std::unique_ptr;

namespace EigenSinn {

  template <typename Scalar>
  using TensorSingleDim = Eigen::Tensor<Scalar, 1>;

  template<typename Scalar, Index Dim>
  inline auto get_broadcast_and_reduction_dims(TensorView<Scalar, Dim> x) {
    // we reduce by all dimensions but the last (channel)
    // and broadcast all but the last
    
    Eigen::array<Index, Dim - 1> reduction_dims;
    Eigen::array<Index, Dim> broadcast_dims;

    for (int i = 0; i < Dim; i++) {
      // channel-first tensor: NCHW, need to skip channel dimension
      if (i != Dim - 1) {
        reduction_dims[i] = i == 0 ? i : i + 1;
      }
      broadcast_dims[i] = x.dimension(i);
    }

    // channel-first format: NCHW
    broadcast_dims[(int)ImageDims::channel] = 1;
    return std::make_tuple(reduction_dims, broadcast_dims);
  }

  // broadcast channel dimension
  template  <typename Scalar, Index Rank, typename Device_>
  inline DeviceTensor<Device_, Scalar, Rank> broadcast_as_last_dim(DeviceTensor<Device_, Scalar, 1>& t, DSizes<Index, Rank> broadcast_dims) {

    DSizes<Index, Rank> reshaped_dims;
    DSizes<Index, Rank> original_dims = broadcast_dims;

    reshaped_dims.fill(1);
    reshaped_dims[(int)ImageDims::channel] = t.dimension(0);
    original_dims[(int)ImageDims::channel] = t.dimension(0);

    DeviceTensor<Device_, Scalar, Rank> broadcasted(original_dims);

    broadcasted.view() = t->reshape(reshaped_dims).broadcast(broadcast_dims);
    return broadcasted;
  }

  // NHWC format
  template<typename Scalar, Index Rank, typename Device_>
  inline auto batch_norm(DeviceTensor<Device_, Scalar, Rank>& x, DeviceTensor<Device_, Scalar, 1>& gamma, DeviceTensor<Device_, Scalar, 1>& beta, float eps, float momentum, 
     DeviceTensor<Device_, Scalar, 1>& running_mean, DeviceTensor<Device_, Scalar, 1>& running_var, bool is_training) {

    DSizes<Index, 1> single_dim{ x->dimension(1) };

    DeviceTensor<Device_, Scalar, 1> mu(single_dim), variance(single_dim), std(single_dim), new_running_var(single_dim), new_running_mean(single_dim);

    DeviceTensor<Device_, Scalar, Rank> mu_broadcasted(x.dimensions()), mean_broadcasted(x.dimensions()), std_broadcasted(x.dimensions());
      
    DeviceTensor<Device_, Scalar, Rank> x_out(x.dimensions()), x_hat(x.dimensions());

    // get sample mean
    DSizes<Index, Rank - 1> reduction_dims;
    DSizes<Index, Rank> broadcast_dims;

    std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims<Scalar, Rank>(*x);

    // if training we compute all the values.
    // otherwise use their running analogs
    if (is_training) {
      // mean
      mu.view() = x->mean(reduction_dims);
      mu_broadcasted = broadcast_as_last_dim(mu, broadcast_dims);

      // variance
      variance.view() = (x - mu_broadcasted)->pow(2.).mean(reduction_dims);

      new_running_mean = momentum * running_mean + (1.0 - momentum) * mu;
      new_running_var = momentum * running_var + (1.0 - momentum) * variance;
      
      std.view() = (*variance + eps).sqrt();
      mean_broadcasted = broadcast_as_last_dim(mu, broadcast_dims);
    }
    else {
      std.view() = (*running_var + eps).sqrt();
      mean_broadcasted = broadcast_as_last_dim(running_mean, broadcast_dims);

      new_running_mean = running_mean;
      new_running_var = running_var;

    }
    std_broadcasted = broadcast_as_last_dim(std, broadcast_dims);

    
    DeviceTensor<Device_, Scalar, Rank> gamma_broadcasted(broadcast_as_last_dim(gamma, broadcast_dims));
    DeviceTensor<Device_, Scalar, Rank> beta_broadcasted(broadcast_as_last_dim(beta, broadcast_dims)) ;

    x_hat = (x - mean_broadcasted) / std_broadcasted;
    x_out = gamma_broadcasted * x_hat + beta_broadcasted;

    return std::make_tuple(x_out, x_hat, new_running_mean, new_running_var, mu, variance);
  }
}