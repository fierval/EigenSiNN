#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, Index Rank>
  class MseLoss : public LossBase<Scalar, Rank> {
  
  public:
    MseLoss() {
      is_dim_set = false;
    }

    void compute(std::any predicted_any, std::any actual_any) override {
      
      Tensor<Scalar, Rank> predicted = from_any<Scalar, Rank>(predicted_any);
      Tensor<Scalar, Rank> actual = from_any<Scalar, Rank>(actual_any);

      array<Index, Rank> predicted_dims = predicted.dimensions();
      array<Index, Rank> actual_dims = actual.dimensions();
      
      if (!is_dim_set) {
        orig_dims = actual.dimensions();
      }

      for (int i = 0; i < Rank; i++) {
        assert(predicted_dims[i] == orig_dims[i]);
      }

      Tensor<Scalar, 0> loss_t = (predicted - actual).pow(2).mean();
      loss = *loss_t.data();

    };

    void backward() {

    }
   
  };
}