#include <iostream>
#include <gtest/gtest.h>
#include <layers/maxpoolinglayer.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Pool2d : public ::testing::Test {

  protected:

    void SetUp() override {

      output.resize(poolDims);
      fakeloss.resize(poolDims);
      dinput.resize(commonData.dims);
      
      commonData.init();

      output.setValues({ { 0.87322980, 0.63184929, 2.51629639},
        {2.07474923, 2.07474923, 1.04950535},
        {-0.23254025, 0.24772950, 0.60365528}} );

      fakeloss.setValues({ {0.31773561, 0.25510252, 0.73881042},
        {0.81441122, 0.74392009, 0.56959468},
        {0.94542354, 0.31825888, 0.96742082} });

      dinput.setValues({ {0.31773561, 0.00000000, 0.25510252, 0.00000000, 0.00000000,
        0.00000000, 0.00000000, 0.73881042},
        {0.00000000, 0.00000000, 0.00000000, 1.55833125, 0.00000000,
        0.56959468, 0.00000000, 0.00000000},
        {0.94542354, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
        0.31825888, 0.00000000, 0.96742082} });
    }

    CommonData2d commonData;
    Tensor<float, 2> output, dinput, fakeloss;
    const array<Index, 1> extents1d = { 4 };
    const array<Index, 2> poolDims = { 3, 3 };

    const int stride = 2;

  };

  TEST_F(Pool2d, Validate) {

    NnTensor<2> t(4, 4);
    t.setConstant(1);

    auto dims = t.dimensions();
    array<Index, 1> extents({ 2 });
    int stride = 2;

    bool res = check_valid_params<2>(extents, stride, dims);

    EXPECT_TRUE(res);
  }

  TEST_F(Pool2d, Forward) {

    MaxPoolingLayer<float, 2> pl(extents1d, stride);
    pl.init();
    pl.forward(commonData.linearInput);

    EXPECT_TRUE(is_elementwise_approx_eq(from_any<float, 2>(pl.get_output()), output));
  }

  TEST_F(Pool2d, Backward) {

    MaxPoolingLayer<float, 2> pl(extents1d, stride);
    pl.init();
    pl.forward(commonData.linearInput);
    pl.backward(commonData.linearInput, fakeloss);

    Tensor<float, 2> deriv = from_any<float, 2>(pl.get_loss_by_input_derivative());
    std::cerr << deriv  << std::endl;

    EXPECT_TRUE(is_elementwise_approx_eq(from_any<float, 2>(pl.get_loss_by_input_derivative()), dinput));
  }


}