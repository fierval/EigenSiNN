#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, typename Actual, Index Rank>
  class MseLoss : public LossBase<Scalar, actual, Rank> {
  
  public:
    MseLoss() {
      is_dim_set = false;
    }

    void step(Tensor<Scalar, Rank>& predicted, Tensor<Actual, Rank>& actual) override {
      
      initialize(predicted, actual);

      predicted_actual_diff = predicted - actual;
      Tensor<Scalar, 0> loss_t = predicted_actual_diff.pow(2).mean();
      loss = *loss_t.data();

      //backward step
      dloss = 2. * spread_grad * predicted_actual_diff;
    }
  };
}