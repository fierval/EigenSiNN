#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, Index Rank>
  class MseLoss : LayerBase {
  
  public:

    void compute(std::any predicted_any, std::any actual_any) override {
      
      Tensor<Scalar, Rank> predicted = from_any<Scalar, Rank>(predicted_any);
      Tensor<Scalar, Rank> actual = from_any<Scalar, Rank>(actual_any);


    };

    void backward() {

    }
  };
}