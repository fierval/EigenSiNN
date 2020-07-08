#pragma once

#include "unsupported/Eigen/CXX11/Tensor"

using namespace Eigen;

namespace EigenSinn {

  template <int Rank>
  using NnTensor = Tensor<float, Rank>;

}