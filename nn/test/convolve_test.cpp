#pragma once

#include "ops/convolutions.hpp"

using namespace EigenSinn;

namespace EigensinnTest {
  TEST(Convolution, Valid) {
    ConvTensor input(5, 4, 4, 2);
    ConvTensor kernel(3, 3, 3, 2);

    input.setConstant(3);
    kernel.setConstant(1);

    ConvTensor output = convolve_valid(input, kernel);
    EXPECT_EQ(output(0, 0, 0, 0), 3 * 9 * 2) << "Failed value test";
    EXPECT_EQ(output.dimension(1), 2) << "Failed dim[1] test";
    EXPECT_EQ(output.dimension(2), 2) << "Failed dim[2] test";
    EXPECT_EQ(output.dimension(3), 3) << "Failed dim[3] test";
  }

  TEST(Convolution, Full) {
    ConvTensor input(5, 4, 4, 2);
    ConvTensor kernel(3, 3, 3, 2);

    input.setConstant(3);
    kernel.setConstant(1);

    ConvTensor output = convolve_full(input, kernel);
    EXPECT_EQ(output(0, 0, 0, 0), 3 * 2) << "Failed value test";
    EXPECT_EQ(output.dimension(1), 6) << "Failed dim[1] test";
    EXPECT_EQ(output.dimension(2), 6) << "Failed dim[2] test";
    EXPECT_EQ(output.dimension(3), 3) << "Failed dim[3] test";
  }

  TEST(Convolution, Same) {
    ConvTensor input(5, 4, 4, 2);
    ConvTensor kernel(3, 3, 3, 2);

    input.setConstant(3);
    kernel.setConstant(1);

    ConvTensor output = convolve_same(input, kernel);
    EXPECT_EQ(output(0, 0, 0, 0), 3 * 4 * 2) << "Failed value test";
    EXPECT_EQ(output.dimension(1), 4) << "Failed dim[1] test";
    EXPECT_EQ(output.dimension(2), 4) << "Failed dim[2] test";
    EXPECT_EQ(output.dimension(3), 3) << "Failed dim[3] test";
  }

} // namespace EigensinnTest