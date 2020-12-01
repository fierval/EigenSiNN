#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, typename Actual, Index Rank = 2, int Layout = ColMajor, typename Device_ = DefaultDevice>
  class MseLoss : public LossBase<Scalar, Actual, Rank, Layout, Device_> {
  
  public:
    MseLoss() {}

    void step(const DeviceTensor<Device_, Scalar, Rank, Layout>& predicted, const DeviceTensor<Device_, Actual, Rank, Layout>& actual) override {
      
      initialize(predicted, actual);

      DeviceTensor<Device_, Scalar, Rank, Layout> predicted_actual_diff(orig_dims);
      predicted_actual_diff.view() = *predicted - actual->cast<Scalar>();

      DeviceTensor<Device_, Scalar, Rank - 1, Layout> loss_t(1);
      loss_t = predicted_actual_diff->pow(2).mean();
      loss = loss_t.to_host()(0);

      //backward step
      dloss = 2. * spread_grad * predicted_actual_diff;
    }
  };
}