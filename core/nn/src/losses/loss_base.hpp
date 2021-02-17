#pragma once

#include <any>
#include <unsupported/Eigen/CXX11/Tensor>
#include <device/device_tensor.hpp>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, typename Actual, Index Rank, int Layout = ColMajor, typename Device_ = DefaultDevice>
  class LossBase {

  public:

    virtual void step(DeviceTensor<Device_, Scalar, Rank, Layout>& predictions_any, DeviceTensor<Device_, Actual, Rank, Layout>& actual_any) = 0;

    virtual Scalar get_output() {
      return loss;
    }

    virtual std::any get_loss_derivative_by_input() {
      return dloss;
    }

    const array<Index, Rank>& get_dims() { return orig_dims; }

  protected:
    // Initializes all sorts of auxiliary dimension values
    inline void initialize(DeviceTensor<Device_, Scalar, Rank, Layout>& predicted, DeviceTensor<Device_, Actual, Rank, Layout>& actual) {

      if (is_initialized) { return; }

      array<Index, Rank> predicted_dims = predicted.dimensions();
      array<Index, Rank> actual_dims = actual.dimensions();

      orig_dims = actual.dimensions();
      dloss.resize(actual_dims);

      // once we reduce only batch dimension is left
      reduced_dims[0] = predicted_dims[0];

      // dimensions reduced by all except batch dimension
      for (int i = 1; i < Rank; i++) { 
        reduction_dims[i - 1] = i; 
      }

      reshape_dims[0] = orig_dims[0];
      broadcast_dims[0] = 1;
      for (int i = 1; i < Rank; i++) {
        reshape_dims[i] = 1;
        broadcast_dims[i] = orig_dims[i];
      }

      spread_grad.resize(orig_dims);

      spread_grad.setConstant(1.);
      for (int i = 0; i < Rank; i++) {
        spread_grad /= static_cast<Scalar>(orig_dims[i]);
      }

      for (int i = 0; i < Rank; i++) {
        assert(predicted_dims[i] == orig_dims[i]);
      }
      is_initialized = true;
    }

    array<Index, Rank> orig_dims;
    array<Index, 1> reduced_dims; // batch dimension only
    array<Index, Rank> reshape_dims; // reshape: first is batch dimension, rest is 1
    array<Index, Rank - 1> reduction_dims; // dimensions, along which we reduce
    array<Index, Rank> broadcast_dims; // broadcast dimensions: same as orig_dims except 1 for batch dimension

    Scalar loss;
    bool is_initialized = false;

    DeviceTensor<Device_, Scalar, Rank, Layout> dloss, spread_grad;
  };
}