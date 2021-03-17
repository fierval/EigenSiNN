#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  // TODO: Add support for any rank
  template<typename Scalar, typename Actual, Index Rank = 2, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class CrossEntropyLoss : public LossBase<Scalar, Actual, Rank, Device_, Layout> {

  public:
    CrossEntropyLoss() {
    }

    void initialize(DeviceTensor<Scalar, Rank, Device_, Layout>& predicted, DeviceTensor<Actual, Rank, Device_, Layout>& actual) {

      if (is_initialized) {
        return;
      }

      LossBase<Scalar, Actual, Rank, Device_, Layout>::initialize(predicted, actual);


      dsum.resize(reduced_dims);

      Scalar cnst = 1;
      for (int i = 0; i < reduced_dims.size(); i++) {
        cnst /= reduced_dims[i];
      }
      dsum.setConstant(cnst);
    }

    void step(DeviceTensor<Scalar, Rank, Device_, Layout>& predicted, DeviceTensor<Actual, Rank, Device_, Layout>& actual) override {

      initialize(predicted, actual);

      DeviceTensor<Scalar, Rank, Device_, Layout> act_scalar(orig_dims);
      act_scalar.view() = actual->template cast<Scalar>();

      // memoize these for the backward pass
      DeviceTensor<Scalar, Rank, Device_, Layout> exp_all(orig_dims);
      exp_all.view() = predicted->exp();

      DeviceTensor<Scalar, Rank - 1, Device_, Layout> exp_sum(reduced_dims);
      exp_sum.view() = exp_all->sum(reduction_dims);

      DeviceTensor<Scalar, 0, Device_, Layout> loss_t;
      loss_t.view() = (-(*predicted * *act_scalar).sum(reduction_dims) + exp_sum->log()).mean();
      loss = loss_t.to_host()(0);

      // backward step
      DeviceTensor<Scalar, Rank, Device_, Layout> dlog(orig_dims);
      dlog.view() = (1. / *exp_sum * *dsum).reshape(reshape_dims).eval().broadcast(broadcast_dims);
      dloss.view() = -1. / orig_dims[0] * *act_scalar + *exp_all * *dlog;
    }

  private:

    DeviceTensor<Scalar, Rank - 1, Device_, Layout> dsum;
  };
}