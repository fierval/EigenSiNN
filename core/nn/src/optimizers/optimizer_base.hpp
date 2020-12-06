#pragma once

#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <layers/layer_base.hpp>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar>
  class OptimizerBase {

  public:
    OptimizerBase(Scalar _lr) 
      : param_set(false)
    , lr(_lr) {

    }
    virtual std::tuple<Scalar *, Scalar *> step(LayerBase<Scalar>& layer) = 0;

  protected:
    // learing rate
    Scalar lr;
    bool param_set;

  };
}