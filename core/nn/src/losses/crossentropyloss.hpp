#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, typename Actual, Index Rank>
  class CrossEntropyLoss : public LossBase<Scalar, Actual, 2> {

  public:
    CrossEntropyLoss() : is_cache_set(false) {
      is_dim_set = false;
    }

    void initialize_and_convert(const Tensor<Scalar, Rank>& predicted_any, const Tensor<Actual, Rank>& actual_any) {

      LossBase::initialize(predicted_any, actual_any);
      if (!is_cache_set) {

        dsum.resize(orig_dims[0]);
        dsum.setConstant(1.);
        dsum = 1. / orig_dims[0] * dsum;

        broadcast_dims[1] = orig_dims[1];
        is_cache_set = true;
      }
    }

    void forward(const Tensor<Scalar, Rank>& predicted, const Tensor<Actual, Rank>& actual) override {

      initialize(predicted, actual);

      // memoize these for the backward pass
      exp_all = predicted.exp();
      exp_sum = exp_all.sum(reduction_dims);

      Tensor<Scalar, 0> loss_t = ((-predicted * actual).sum(reduction_dims) + exp_sum.log()).mean();
      loss = loss_t(0);
    };

    void backward() {

      Tensor<Scalar, 2> dlog = (1. / exp_sum * dsum).reshape(array<Index, 2>{orig_dims[0], 1}).eval().broadcast(broadcast_dims);
      dloss = -1. / orig_dims[0] * actual + exp_all * dlog;
    }

  private:
    Tensor<Scalar, 2> exp_all, actual;
    Tensor<Scalar, 1> exp_sum, dsum;
    bool is_cache_set;
    array<Index, 2> broadcast_dims = { 1, 0 };
    array<Index, 1> reduction_dims = { 1 };

  };
}