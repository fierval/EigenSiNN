#pragma once

#include "ops/opsbase.hpp"

namespace EigenSinn {
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class Conv2d {};
}