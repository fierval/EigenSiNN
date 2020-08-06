#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include <any>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar>
  class OptimizerBase {

  public:
    virtual void step(std::any weights) = 0;
    virtual void step(std::any weights, std::any bias) {};

  protected:
    // learing rate
    Scalar lr;
  };
}