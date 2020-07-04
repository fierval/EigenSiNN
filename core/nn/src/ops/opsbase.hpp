#pragma once

#include "unsupported/Eigen/CXX11/Tensor"

namespace EigenSinn {
  using namespace Eigen;

  template <int Rank>
  using NnTensor = Tensor<float, Rank>;

}