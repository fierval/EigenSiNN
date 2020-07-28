#pragma once

#include "ops/opsbase.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  template<typename Scalar, Index Rank>
  void out_compare(const Tensor<Scalar, Rank>& expected, const Tensor<Scalar, Rank>& target) {
    std::cerr << "Expected" << std::endl << "==========" << std::endl << std::endl;
    std::cerr << expected << std::endl << std::endl << "Actual: \n========" << std::endl << std::endl;
    std::cerr << target << std::endl << std::endl << "################################################" << std::endl << std::endl;
  }

}