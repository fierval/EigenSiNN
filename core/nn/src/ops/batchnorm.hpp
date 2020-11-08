#pragma once

#include "opsbase.hpp"
#include <device/device_tensor.hpp>
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

  // broadcast channel dimension - first in the list of arguments, las
  template  <typename Scalar, Index Rank, typename Device_>
  inline TensorView<Scalar, Rank> broadcast_as_last_dim(const TensorView<Scalar, 1>& t, DSizes<Index, Rank> broadcast_dims, Device_& device) {

    DSizes<Index, Rank> reshaped_dims;
    DSizes<Index, Rank> original_dims = broadcast_dims;

    reshaped_dims.fill(1);
    reshaped_dims[(int)ImageDims::channel] = t.dimension(0);
    original_dims[(int)ImageDims::channel] = t.dimension(0);

    TensorView<Scalar, Rank> broadcasted(create_device_view<Device_, Scalar, Rank>(original_dims, device));

    broadcasted.device(device) = t.reshape(reshaped_dims).broadcast(broadcast_dims);
    return broadcasted;
  }

  // NHWC format
  template<typename Scalar, Index Rank, typename Device_>
  inline auto batch_norm(TensorView<Scalar, Rank>& x, TensorView<Scalar, 1>& gamma, TensorView<Scalar, 1>& beta, float eps, float momentum, 
     TensorView<Scalar, 1>& running_mean, TensorView<Scalar, 1>& running_var, bool is_training, Device_& device) {

    DSizes<Index, 1> single_dim{ x.dimension(1) };

    TensorView<Scalar, 1> mu(create_device_view<Device_, Scalar, 1>(single_dim, device)),
      variance(create_device_view<Device_, Scalar, 1>(single_dim, device)), std(create_device_view<Device_, Scalar, 1>(single_dim, device)),
      new_running_var(create_device_view<Device_, Scalar, 1>(single_dim, device)), new_running_mean(create_device_view<Device_, Scalar, 1>(single_dim, device));

    TensorView<Scalar, Rank> mu_broadcasted(create_device_view<Device_, Scalar, Rank>(x.dimensions(), device)),
      mean_broadcasted(create_device_view<Device_, Scalar, Rank>(x.dimensions(), device)),
      std_broadcasted(create_device_view<Device_, Scalar, Rank>(x.dimensions(), device));
      
    PtrTensorView<Scalar, Rank> x_out(create_device_ptr<Device_, Scalar, Rank>(x.dimensions(), device)),
      x_hat(create_device_ptr<Device_, Scalar, Rank>(x.dimensions(), device));

    // get sample mean
    DSizes<Index, Rank - 1> reduction_dims;
    DSizes<Index, Rank> broadcast_dims;

    std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims<Scalar, Rank>(x);

    // if training we compute all the values.
    // otherwise use their running analogs
    if (is_training) {
      // mean
      mu.device(device) = x.mean(reduction_dims);
      mu_broadcasted.device(device) = broadcast_as_last_dim(mu, broadcast_dims, device);

      // variance
      variance.device(device) = (x - mu_broadcasted).pow(2.).mean(reduction_dims);

      new_running_mean.device(device) = momentum * running_mean + (1.0 - momentum) * mu;
      new_running_var.device(device) = momentum * running_var + (1.0 - momentum) * variance;
      
      std.device(device) = (variance + eps).sqrt();
      mean_broadcasted = broadcast_as_last_dim<Scalar, Rank>(mu, broadcast_dims, device);
    }
    else {
      std.device(device) = (running_var + eps).sqrt();
      mean_broadcasted = broadcast_as_last_dim<Scalar, Rank>(running_mean, broadcast_dims, device);

      move_to(new_running_mean, running_mean, device);
      move_to(new_running_var, running_var, device);

    }
    std_broadcasted = broadcast_as_last_dim<Scalar, Rank>(std, broadcast_dims, device);

    
    TensorView<Scalar, Rank> gamma_broadcasted(broadcast_as_last_dim<Scalar, Rank>(gamma, broadcast_dims, device));
    TensorView<Scalar, Rank> beta_broadcasted(broadcast_as_last_dim<Scalar, Rank>(beta, broadcast_dims, device));

    x_hat->device(device) = (x - mean_broadcasted) / std_broadcasted;
    x_out->device(device) = gamma_broadcasted * *x_hat + beta_broadcasted;

    free(beta_broadcasted, device);
    free(gamma_broadcasted, device);
    free(mu_broadcasted, device);
    free(mean_broadcasted, device);
    free(std, device);

    return std::make_tuple(std::move(x_out), std::move(x_hat), new_running_mean, new_running_var, mu, variance);
  }
}