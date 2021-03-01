#include <iostream>
#include <gtest/gtest.h>
#include <layers/maxpooling.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class Pool2dGpu : public ::testing::Test {

  protected:

    void SetUp() override {

      output.resize(poolDims);
      fakeloss.resize(poolDims);
      dinput.resize(cd.dims);
      
      cd.init();

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

    CommonData2d<GpuDevice> cd;
    DeviceTensor<float, 2, GpuDevice> output, dinput, fakeloss;
    const array<Index, 1> extents1d = { 4 };
    const array<Index, 2> poolDims = { 3, 3 };

    const int stride = 2;

  };

  TEST_F(Pool2dGpu, Forward) {
    Input<float, 2, GpuDevice> input;

    input.set_input(cd.linearInput);

    MaxPooling<float, 2, GpuDevice> pl(extents1d, stride);
    pl.init();
    pl.forward(input);

    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_output(), output));
  }

  TEST_F(Pool2dGpu, Backward) {

    Input<float, 2, GpuDevice> input;
    input.set_input(cd.linearInput);

    MaxPooling<float, 2, GpuDevice> pl(extents1d, stride);
    pl.init();
    pl.forward(input);

    pl.backward(input, fakeloss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_loss_by_input_derivative(), dinput));
  }
}