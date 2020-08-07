#pragma once

#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar>
  class OptimizerBase {

  public:
    virtual std::tuple<std::any, std::any> step(const std::any weights_any, const std::any bias_any, const std::any dweights_any, const std::any dbias_any) = 0;

  protected:
    // learing rate
    Scalar lr;
  };
}