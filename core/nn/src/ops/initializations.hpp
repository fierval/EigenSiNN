#pragma once

#include "device/device_tensor.hpp"

namespace EigenSinn {


  template<typename Scalar, Index Rank>
  inline Scalar get_conv_std(array<Index, Rank> layer_dims) {

    assert(Rank == 4);
    Tensor<Scalar, 0> std;
    std.setConstant(1. / (layer_dims[1] * layer_dims[2] * layer_dims[3]));
    std = std.sqrt();

    return std(0);

  }

  template<typename Scalar, Index Rank>
  inline Scalar get_fc_std(array<Index, Rank> layer_dims) {

    assert(Rank == 2);
    Tensor<Scalar, 0> std;
    std.setConstant(1. / layer_dims[0]);
    std = std.sqrt();

    return std(0);

  }

  template <typename Scalar, Index Rank, int Layout = RowMajor, typename Device_ = ThreadPoolDevice>
  inline DeviceTensor<Scalar, Rank, Device_, Layout> generate_xavier(DSizes<Index, Rank> layer_dims) {

    assert(Rank == 2 || Rank == 4);

    // all this wrapping due to possible GPU invokations
    Scalar std;

    switch (Rank) {
    case 4:
      std = get_conv_std<Scalar, Rank>(layer_dims);
      break;
    case 2:
      std = get_fc_std<Scalar, Rank>(layer_dims);
      break;
    default:
      assert(false);
    }

    DeviceTensor<Scalar, Rank, Device_, Layout> weights(layer_dims);
    weights.template setRandom<::internal::NormalRandomGenerator<Scalar>>();

    weights.view() = std * *weights;
    return weights;
  }
}