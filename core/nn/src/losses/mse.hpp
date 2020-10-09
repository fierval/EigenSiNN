#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, typename Actual, Index Rank>
  class MseLoss : public LossBase<Scalar, Actual, Rank> {
  
  public:
    MseLoss() {
      is_dim_set = false;
    }

    void step(const Tensor<Scalar, Rank>& predicted, const Tensor<Actual, Rank>& actual) override {
      
      initialize(predicted, actual);

      Tensor<float, Rank> predicted_actual_diff = predicted - actual.cast<Scalar>();
      Tensor<Scalar, 0> loss_t = predicted_actual_diff.pow(2).mean();
      loss = *loss_t.data();

      //backward step
      dloss = 2. * spread_grad * predicted_actual_diff;
    }
  };
}