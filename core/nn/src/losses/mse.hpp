#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, typename Actual>
  class MseLoss : public LossBase<Scalar, 2> {
  
  public:
    MseLoss() {
      is_dim_set = false;
    }

    void forward(const Tensor<Scalar, Rank>& predicted_any, const Tensor<Actual, Rank>& actual_any) override {
      
      initialize(predicted, actual);

      predicted_actual_diff = predicted - actual;
      Tensor<Scalar, 0> loss_t = predicted_actual_diff.pow(2).mean();
      loss = *loss_t.data();

    };

    void backward() {
      dloss = 2. * spread_grad * predicted_actual_diff;
    }
   
  private:
    Tensor<Scalar, 2> predicted_actual_diff;

  };
}