#pragma once

#include <any>
#include <unsupported/Eigen/CXX11/Tensor>
#include <device/device_tensor.hpp>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, typename Actual, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class LossBase {

  public:

    virtual void step(TensorAdapter<Scalar, Device_>& predictions_any, TensorAdapter<Actual, Device_>& actual_any) = 0;

    virtual Scalar get_output() {
      return loss;
    }

    virtual PtrTensorAdapter<Scalar, Device_> get_loss_derivative_by_input() {
      return dloss.raw();
    }

    const array<Index, Rank>& get_dims() { return orig_dims; }

  protected:
    // Initializes all sorts of auxiliary dimension values
    inline void initialize(DeviceTensor<Scalar, Rank, Device_, Layout>& predicted, DeviceTensor<Actual, Rank, Device_, Layout>& actual) {

      if (is_initialized) { return; }

      orig_dims = actual.dimensions();
      dloss.resize(orig_dims);

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
        spread_grad.view() = *spread_grad / static_cast<Scalar>(orig_dims[i]);
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

    DeviceTensor<Scalar, Rank, Device_, Layout> dloss, spread_grad;
  };
}