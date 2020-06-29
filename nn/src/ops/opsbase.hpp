#pragma once

#include "unsupported/Eigen/CXX11/Tensor"

namespace EigenSinn {

  template <int Rank>
  using NnTensor = Eigen::Tensor<float, Rank>;

}