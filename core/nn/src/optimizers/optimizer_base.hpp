#pragma once

#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <layers/layer_base.hpp>
#include <device/device_tensor.hpp>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, int Layer = ColMajor, typename Device_ = DefaultDevice>
  class OptimizerBase {
    
  public:
    typedef std::tuple<DeviceTensor<Device_, Scalar, Index, Rank, Layout>, DeviceTensor<Device_, Scalar, Index, 1, Layout>> DeviceWeightBiasTuple;

    OptimizerBase(Scalar _lr) 
      : param_set(false)
    , lr(_lr) {

    }
    virtual DeviceTensorTuple step(LayerBase<Scalar>& layer) = 0;

  protected:
    // learing rate
    Scalar lr;
    bool param_set;

  };
}