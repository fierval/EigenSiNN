#include <iostream>
#include <gtest/gtest.h>
#include <layers/maxpooling.hpp>
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class Pool4dGpu : public ::testing::Test {

  protected:

    void SetUp() override {

      cd.init();
      output.resize(cd.poolDims);
      fakeloss.resize(cd.poolDims);

      output.setValues({ {{{0.95949411, 0.85607505},
        {0.98773926, 0.62964255}},

        {{0.96129739, 0.72482306},
        {0.63725311, 0.87211448}},

        {{0.85018283, 0.58271384},
        {0.98281318, 0.96604007}} },


        {{{0.84584051, 0.74516875},
        {0.84589291, 0.96158671}},

        {{0.84837216, 0.98481309},
        {0.78823119, 0.77357787}},

        {{0.92674822, 0.82311010},
        {0.86051500, 0.99673647}} } });

      fakeloss.setValues({ {{{0.53762484, 0.08841801},
        {0.26598877, 0.08771902}},

        {{0.14441812, 0.17294925},
        {0.08538872, 0.09313697}},

        {{0.27319407, 0.67958736},
        {0.58928680, 0.76245397}} },


        {{{0.93036085, 0.36994016},
        {0.41254914, 0.93899775}},

        {{0.51614463, 0.66111606},
        {0.01932418, 0.13548207}},

        {{0.78922635, 0.48408800},
        {0.61365259, 0.99544948}} } });

    
      dinput.resize(cd.dims);
      dinput.setValues({ {{{0.00000000, 0.53762484, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 0.08841801},
        {0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.26598877, 0.08771902, 0.00000000}},

        {{0.00000000, 0.00000000, 0.00000000, 0.17294925},
        {0.00000000, 0.14441812, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.08538872, 0.00000000, 0.00000000, 0.09313697}},

        {{0.27319407, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 0.67958736},
        {0.00000000, 0.00000000, 0.76245397, 0.00000000},
        {0.00000000, 0.58928680, 0.00000000, 0.00000000}} },


        {{{0.00000000, 0.93036085, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.36994016, 0.00000000},
        {0.00000000, 0.41254914, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.93899775, 0.00000000}},

        {{0.00000000, 0.51614463, 0.00000000, 0.66111606},
        {0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.01932418, 0.00000000, 0.13548207}},

        {{0.78922635, 0.00000000, 0.48408800, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 0.00000000},
        {0.00000000, 0.00000000, 0.00000000, 0.99544948},
        {0.00000000, 0.61365259, 0.00000000, 0.00000000}} } });

    }

    CommonData4d<GpuDevice, RowMajor> cd;
    DeviceTensor<float, 4, GpuDevice, RowMajor> output, dinput, fakeloss;
    const DSizes<Index, 2> extents2d{ 2, 2 };

    const int stride = 2;

  };

  TEST_F(Pool4dGpu, Forward) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    MaxPooling<float, 4, GpuDevice, RowMajor> pl(extents2d, stride);

    pl.init();
    pl.forward(input.get_output());

    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_output(), output));
  }

  TEST_F(Pool4dGpu, Backward) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    MaxPooling<float, 4, GpuDevice, RowMajor> pl(extents2d, stride);

    pl.init();
    pl.forward(input.get_output());
    pl.backward(input.get_output(), fakeloss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_output(), output));
    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_loss_by_input_derivative(), dinput));
  }

  TEST_F(Pool4dGpu, BackwardCudnn) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    MaxPooling<float, 4, GpuDevice, RowMajor> pl(extents2d, stride);
    
    pl.set_cudnn(true);
    pl.init();
    pl.forward(input.get_output());
    pl.backward(input.get_output(), fakeloss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_output(), output));
    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_loss_by_input_derivative(), dinput));
  }


}