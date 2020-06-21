#include "ops/convolutions.hpp"
#include "layers/input.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace EigenSinn;

namespace EigenSinnTest {
  TEST(Reverse, Reverse) {

    Eigen::Tensor<int, 2> a(4, 3);

    a.setValues({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12} });

    Eigen::array<bool, 2> rev_index({ true, false });
    a.reverse(rev_index);

    std::cerr << a;
  }

}