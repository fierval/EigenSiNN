
#include "ops/convolutions.hpp"
#include <gtest/gtest.h>

using namespace EigenSinn;
using ConvTensor = Tensor<float, 4>;

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

  TEST(Convolution, SameEqPadding) {
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

  TEST(Convolution, SameNonEqPadding) {
    ConvTensor input(5, 6, 6, 2);
    ConvTensor kernel(3, 4, 4, 2);

    input.setConstant(3);
    kernel.setConstant(1);

    ConvTensor output = convolve_same(input, kernel);
    EXPECT_EQ(output(0, 0, 0, 0), 3 * 4 * 2) << "Failed value test";
    EXPECT_EQ(output.dimension(1), 6) << "Failed dim[1] test";
    EXPECT_EQ(output.dimension(2), 6) << "Failed dim[2] test";
    EXPECT_EQ(output.dimension(3), 3) << "Failed dim[3] test";
  }

  TEST(Convolution, ComputeDimensions) {

    ConvTensor input(5, 8, 8, 2);
    ConvTensor kernel(4, 3, 3, 2);

    auto dims = get_output_dimensions(input, kernel, ConvType::valid);

    EXPECT_EQ(dims[0], 5) << "Failed valid dims[0]";
    EXPECT_EQ(dims[1], 6) << "Failed valid dims[1]";
    EXPECT_EQ(dims[2], 6) << "Failed valid dims[2]";
    EXPECT_EQ(dims[3], 4) << "Failed valid dims[3]";

    dims = get_output_dimensions(input, kernel, ConvType::full);

    EXPECT_EQ(dims[1], 10) << "Failed full dims[1]";
    EXPECT_EQ(dims[2], 10) << "Failed full dims[2]";

    dims = get_output_dimensions(input, kernel, ConvType::same);

    EXPECT_EQ(dims[1], 8) << "Failed same dims[1]";
    EXPECT_EQ(dims[2], 8) << "Failed same dims[2]";
  }
} // namespace EigensinnTest