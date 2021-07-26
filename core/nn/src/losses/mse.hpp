#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, typename Actual, Index Rank = 2, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class MseLoss : public LossBase<Scalar, Actual, Rank, Device_, Layout> {
  
  public:
    MseLoss() {}

    void step(PtrTensorAdapter<Scalar, Device_>& predicted_adapter, PtrTensorAdapter<Actual, Device_>& actual_adapter) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> predicted(predicted_adapter);
      DeviceTensor<Actual, Rank, Device_, Layout> actual(actual_adapter);

      initialize(predicted, actual);

      DeviceTensor<Scalar, Rank, Device_, Layout> predicted_actual_diff(orig_dims);
      predicted_actual_diff.view() = *predicted - actual->cast<Scalar>();

      DeviceTensor<Scalar, 0, Device_, Layout> loss_t;
      loss_t.view() = predicted_actual_diff->pow(2).mean();
      loss = loss_t.to_host()(0);

      //backward step
      dloss.view() = 2. * *spread_grad * *predicted_actual_diff;
    }
  };
}