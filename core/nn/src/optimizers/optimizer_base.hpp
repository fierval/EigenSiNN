#pragma once

#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <layers/layer_base.hpp>
#include <device/device_tensor.hpp>

using namespace Eigen;

enum Optimizers : int{
  Adam,
  SGD
};

namespace EigenSinn {

  template<typename Scalar, typename Device_>
  using DeviceWeightBiasTuple = std::tuple<PtrTensorAdapter<Scalar, Device_>, PtrTensorAdapter<Scalar, Device_>>;

  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class OptimizerBase {


  public:
    
    OptimizerBase(float _lr) 
      : param_set(false)
    , lr(_lr) {

    }

    virtual DeviceWeightBiasTuple<Scalar, Device_> step(LayerBase<Scalar, Device_>& ) = 0;

  protected:
    // learing rate
    float lr;
    bool param_set;

  };
}