#pragma once

#include "optimizer_base.hpp"

namespace EigenSinn {

  template <typename Scalar, Index Rank>
  class SGD : public OptimizerBase<Scalar> {

    SGD(Scalar _lr) : lr(_lr) {

    }

  };
}