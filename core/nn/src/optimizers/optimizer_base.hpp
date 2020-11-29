#pragma once

#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <layers/layer_base.hpp>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, typename Device_ = DefaultDevice>
  class OptimizerBase {

  public:
    OptimizerBase(Scalar _lr, Dispatcher<Device_>& _dispatcher) 
      : param_set(false)
    , lr(_lr)
    , dispatcher(_dispatcher) {

    }
    virtual std::tuple<Scalar *, Scalar *> step(LayerBase<Scalar>& layer) = 0;

    inline static Dispatcher<DefaultDevice> default_dispatcher = Dispatcher<DefaultDevice>();

  protected:
    // learing rate
    Scalar lr;
    bool param_set;

    Dispatcher<Device_>& dispatcher;

  };
}