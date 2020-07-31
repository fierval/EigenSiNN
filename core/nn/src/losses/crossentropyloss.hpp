#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, Index Rank>
  class CrossEntropyLoss : public LossBase<Scalar, Rank> {
  
  public:
    CrossEntropyLoss() {
      is_dim_set = false;
    }

    void forward(std::any predicted_any, std::any actual_any) override {
      
      auto tensors = initialize_and_convert(predicted_any, actual_any);

      Tensor<Scalar, Rank> predicted = tensors.first;
      Tensor<Scalar, Rank> actual = tensors.second;

      predicted_actual_diff = predicted - actual;
      Tensor<Scalar, 0> loss_t = predicted_actual_diff.pow(2).mean();
      loss = *loss_t.data();

    };

    void backward() {
      dloss = 2. * spread_grad * predicted_actual_diff;
    }
   
  private:

    Tensor<Scalar, Rank> predicted_actual_diff;
  };
}