#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, Index Rank>
  class MseLoss : LossBase<Scalar, Rank> {
  
  public:

    void compute(std::any predicted_any, std::any actual_any) override {
      
      Tensor<Scalar, Rank> predicted = from_any<Scalar, Rank>(predicted_any);
      Tensor<Scalar, Rank> actual = from_any<Scalar, Rank>(actual_any);

      array<Index, Rank> predicted_dims = predicted.dimensions();
      array<Index, Rank> actual_dims = actual.dimensions();

      for (int i = 0; i < Rank; i++) {
        assert(actual_dims[i] == orig_dims[i]);
      }

      loss = (predicted - actual).pow(2).mean();

    };

    void backward() {

    }
  };
}