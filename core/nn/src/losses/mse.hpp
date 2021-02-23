#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, typename Actual, Index Rank = 2, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class MseLoss : public LossBase<Scalar, Actual, Rank, Layout, Device_> {
  
  public:
    MseLoss() {}

    void step(DeviceTensor<Scalar, Rank, Device_, Layout>& predicted, DeviceTensor<Actual, Rank, Device_, Layout>& actual) override {
      
      initialize(predicted, actual);

      DeviceTensor<Scalar, Rank, Device_, Layout> predicted_actual_diff(orig_dims);
      predicted_actual_diff.view() = *predicted - actual->cast<Scalar>();

      DeviceTensor<Device_, Scalar, 0, Layout> loss_t;
      loss_t.view() = predicted_actual_diff->pow(2).mean();
      loss = loss_t.to_host()(0);

      //backward step
      dloss.view() = 2. * *spread_grad * *predicted_actual_diff;
    }
  };
}