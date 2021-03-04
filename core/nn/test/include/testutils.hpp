#pragma once

#include "ops/opsbase.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  template<typename Scalar, Index Rank, int Layout = ColMajor>
  void out_compare(const Tensor<Scalar, Rank, Layout>& expected, const Tensor<Scalar, Rank, Layout>& target) {
    std::cerr << "Expected" << std::endl << "==========" << std::endl << std::endl;
    std::cerr << expected << std::endl << std::endl << "Actual: \n========" << std::endl << std::endl;
    std::cerr << target << std::endl << std::endl << "################################################" << std::endl << std::endl;
  }

}