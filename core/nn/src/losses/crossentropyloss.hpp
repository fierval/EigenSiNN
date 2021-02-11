#pragma once

#include "ops/conversions.hpp"
#include "loss_base.hpp"

namespace EigenSinn {

  // TODO: Add support for any rank
  template<typename Scalar, typename Actual, Index Rank = 2, int Layout = ColMajor, typename Device_ = DefaultDevice>
  class CrossEntropyLoss : public LossBase<Scalar, Actual, Rank, Layout, Device_> {

  public:
    CrossEntropyLoss() {
    }

    void initialize(DeviceTensor<Device_, Scalar, Rank, Layout>& predicted, DeviceTensor<Device_, Actual, Rank, Layout> actual) {

      if (is_initialized) {
        return;
      }

      LossBase::initialize(predicted, actual);


      dsum.resize(reduced_dims);

      Scalar cnst = 1;
      for (int i = 0; i < reduced_dims.size(); i++) {
        cnst /= reduced_dims[i];
      }
      dsum.setConstant(cnst);
    }

    void step(DeviceTensor<Device_, Scalar, Rank, Layout>& predicted, DeviceTensor<Device_, Actual, Rank, Layout>& actual) override {

      initialize(predicted, actual);

      DeviceTensor<Device_, Scalar, Rank, Layout> act_scalar(orig_dims);
      act_scalar.view() = actual->cast<Scalar>();

      // memoize these for the backward pass
      DeviceTensor<Device_, Scalar, Rank, Layout> exp_all(orig_dims);
      exp_all.view() = predicted->exp();

      DeviceTensor<Device_, Scalar, Rank - 1, Layout> exp_sum(reduced_dims);
      exp_sum.view() = exp_all->sum(reduction_dims);

      DeviceTensor<Device_, Scalar, 0, Layout> loss_t;
      loss_t.view() = (-(predicted * act_scalar)->sum(reduction_dims) + exp_sum->log()).mean();
      loss = loss_t.to_host()(0);

      // backward step
      DeviceTensor<Device_, Scalar, Rank, Layout> dlog(orig_dims);
      dlog.view() = (1. / *exp_sum * *dsum).reshape(reshape_dims).eval().broadcast(broadcast_dims);
      dloss = -1. / orig_dims[0] * act_scalar + exp_all * dlog;
    }

  private:

    DeviceTensor<Device_, Scalar, Rank - 1, Layout> dsum;
  };
}