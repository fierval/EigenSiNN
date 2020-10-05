#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  template<typename Scalar, typename Actual, Index Rank>
  class CrossEntropyLoss : public LossBase<Scalar, Actual, Rank> {

  public:
    CrossEntropyLoss() : is_cache_set(false) {
      is_dim_set = false;
    }

    void initialize(const Tensor<Scalar, Rank>& predicted, const Tensor<Actual, Rank> actual) {

      if (!is_cache_set) {
        LossBase::initialize(predicted, actual);

        dsum.resize(orig_dims[0]);
        dsum.setConstant(1. / orig_dims[0]);

        broadcast_dims[1] = orig_dims[1];
        is_cache_set = true;
      }
    }

    void step(const Tensor<Scalar, Rank>& predicted, const Tensor<Actual, Rank>& actual) override {

      initialize(predicted, actual);

      // memoize these for the backward pass
      Tensor<Scalar, Rank> exp_all = predicted.exp();
      Tensor<Scalar, 1> exp_sum = exp_all.sum(reduction_dims);

      Tensor<Scalar, 0> loss_t = ((-predicted * actual).sum(reduction_dims) + exp_sum.log()).mean();
      loss = loss_t(0);
      
      // backward step
      Tensor<Scalar, Rank> dlog = (1. / exp_sum * dsum).reshape(array<Index, 2>{orig_dims[0], 1}).eval().broadcast(broadcast_dims);
      dloss = -1. / orig_dims[0] * actual + exp_all * dlog;
    }

  private:

    Tensor<Scalar, 1> dsum;
    bool is_cache_set;
    array<Index, 2> broadcast_dims = { 1, 0 };
    array<Index, 1> reduction_dims = { 1 };

  };
}