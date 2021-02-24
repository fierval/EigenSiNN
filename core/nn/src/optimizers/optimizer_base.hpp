#pragma once

#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <layers/layer_base.hpp>
#include <device/device_tensor.hpp>

using namespace Eigen;

namespace EigenSinn {

  typedef std::tuple<std::any, std::any> DeviceWeightBiasTuple;

  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class OptimizerBase {

  public:
    
    OptimizerBase(Scalar _lr) 
      : param_set(false)
    , lr(_lr) {

    }

    virtual DeviceWeightBiasTuple step(LayerBase<Scalar>& ) = 0;

  protected:
    // learing rate
    Scalar lr;
    bool param_set;

  };
}