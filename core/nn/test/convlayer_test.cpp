#include <gtest/gtest.h>
#include <iostream>
#include "layers/convolution.hpp"
#include "include/commondata4d.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Convolution : public ::testing::Test {

  protected:
    void SetUp() {

    }

    Tensor<float, 4> fakeloss;
  };

  TEST_F(Convolution, Backprop) {

    Eigen::Tensor<int, 2> a(4, 3);

    a.setValues({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12} });

    Eigen::array<bool, 2> rev_index({ true, false });
    Eigen::Tensor<int, 2> b = a.reverse(rev_index);

    EXPECT_EQ(b(0, 0), 10);
  }

}