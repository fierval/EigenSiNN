#include <iostream>
#include <gtest/gtest.h>
#include <layers/maxpooling.hpp>
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class Pool4d : public ::testing::Test {

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

    void SetUpStride1() {
      dinput.setValues({ {{{0.00000000, 0.84976017, 0.00000000, 0.00000000},
                  {0.00000000, 0.83918720, 0.00000000, 0.83708912},
                  {0.68415672, 0.00000000, 0.00000000, 0.00000000},
                  {0.00000000, 1.02657342, 0.01823634, 0.00000000}},

                 {{0.00000000, 0.00000000, 0.00000000, 0.18182540},
                  {0.00000000, 0.63309002, 0.00000000, 0.00000000},
                  {0.00000000, 0.00000000, 0.00000000, 0.48198962},
                  {0.79064542, 0.00000000, 0.13256913, 0.77111048}},

                 {{0.87654692, 0.81638652, 0.00000000, 0.00000000},
                  {0.03452933, 0.00000000, 0.00000000, 0.65065503},
                  {0.00000000, 0.00000000, 1.14110994, 0.00000000},
                  {0.00000000, 1.58678460, 0.00000000, 0.00000000}}},


                {{{0.00000000, 1.70723736, 0.00000000, 0.00000000},
                  {0.00000000, 0.00000000, 0.24067771, 0.00000000},
                  {0.00000000, 1.23908710, 0.00000000, 0.17910063},
                  {0.00000000, 0.00000000, 0.82995802, 0.00000000}},

                 {{0.00000000, 0.39639467, 0.00000000, 0.50150609},
                  {0.00000000, 0.00000000, 1.07137978, 0.00000000},
                  {0.00000000, 0.03796721, 0.00000000, 0.00000000},
                  {0.00000000, 0.91705418, 0.00000000, 0.35360551}},

                 {{0.28318352, 0.00000000, 1.13956189, 0.00000000},
                  {0.00000000, 0.10765439, 0.00000000, 0.00000000},
                  {0.00000000, 0.00000000, 1.09240949, 0.95988500},
                  {0.00000000, 0.37381238, 0.00000000, 0.00000000}}} });


      fakeloss.resize(2, 3, 3, 3);
      fakeloss.setValues({ {{{0.18871814, 0.66104203, 0.67412460},
                  {0.68415672, 0.83918720, 0.16296452},
                  {0.57792860, 0.44864488, 0.01823634}},

                 {{0.04834914, 0.10648906, 0.18182540},
                  {0.43325162, 0.04500020, 0.48198962},
                  {0.79064542, 0.13256913, 0.77111048}},

                 {{0.87654692, 0.81638652, 0.65065503},
                  {0.03452933, 0.64253539, 0.03129411},
                  {0.71614343, 0.87064117, 0.46728039}}},


                {{{0.93335724, 0.77388012, 0.24067771},
                  {0.24605733, 0.29329503, 0.17910063},
                  {0.69973481, 0.71793348, 0.11202455}},

                 {{0.39639467, 0.08519226, 0.50150609},
                  {0.03796721, 0.32585174, 0.66033578},
                  {0.68264580, 0.23440838, 0.35360551}},

                 {{0.28318352, 0.95920134, 0.18036056},
                  {0.10765439, 0.64352757, 0.95874316},
                  {0.37381238, 0.44888192, 0.00114185}}} });

      output.resize(2, 3, 3, 3);
      output.setValues({ {{{0.95949411, 0.95949411, 0.85607505},
            {0.90876633, 0.73222357, 0.85607505},
            {0.98773926, 0.98773926, 0.62964255}},

           {{0.96129739, 0.96129739, 0.72482306},
            {0.96129739, 0.96129739, 0.82289129},
            {0.63725311, 0.54370487, 0.87211448}},

           {{0.85018283, 0.75999165, 0.58271384},
            {0.74964321, 0.96604007, 0.96604007},
            {0.98281318, 0.98281318, 0.96604007}}},


          {{{0.84584051, 0.84584051, 0.74516875},
            {0.84589291, 0.84589291, 0.93618810},
            {0.84589291, 0.96158671, 0.96158671}},

           {{0.84837216, 0.87681556, 0.98481309},
            {0.66477776, 0.87681556, 0.87681556},
            {0.78823119, 0.78823119, 0.77357787}},

           {{0.92674822, 0.82311010, 0.82311010},
            {0.64998370, 0.96253598, 0.99673647},
            {0.86051500, 0.96253598, 0.99673647}}} });

    }
    CommonData4d<ThreadPoolDevice> cd;
    DeviceTensor<float, 4> output, dinput, fakeloss;
    const DSizes<Index, 2> extents2d{ 2, 2 };
    const DSizes<Index, 1> extents1d{ 2 };

    const int stride = 2;

  };

  TEST_F(Pool4d, Validate) {

    DeviceTensor<float, 4> t(1, 3, 4, 4);
    t.setConstant(1);

    auto dims = t.dimensions();
    DSizes<Index, 2> extents{ 2, 2 };
    int stride = 2;

    bool res = check_valid_params<4>(extents, stride, dims);

    EXPECT_TRUE(res);
  }

  TEST_F(Pool4d, BadExtent) {

    DeviceTensor<float, 2> t(4, 4);
    t.setConstant(1);

    auto dims = t.dimensions();
    DSizes<Index, 1> extents({ 5 });
    int stride = 2;

    bool res = check_valid_params<2>(extents, stride, dims);

    EXPECT_FALSE(res);
  }

  TEST_F(Pool4d, BadStride4d) {

    DeviceTensor<float, 4> t(3, 4, 10, 10);
    t.setConstant(1);

    auto dims = t.dimensions();
    DSizes<Index, 2> extents{ 3, 3 };
    int stride = 2;

    bool res = check_valid_params<4>(extents, stride, dims);

    EXPECT_FALSE(res);
  }

  TEST_F(Pool4d, Backward) {

    Input<float> input;
    input.set_input(cd.convInput);

    MaxPooling<float, 4> pl(extents2d, stride);

    pl.init();
    pl.forward(input);
    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_output(), output));

    pl.backward(input, fakeloss.raw());
    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_loss_by_input_derivative(), dinput));
  }

  TEST_F(Pool4d, Forward) {

    Input<float> input;
    input.set_input(cd.convInput);

    MaxPooling<float, 4> pl(extents2d, stride);

    pl.init();
    pl.forward(input);

    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_output(), output));
  }

  TEST_F(Pool4d, BackwardStride1) {

    Input<float> input;
    input.set_input(cd.convInput);

    SetUpStride1();

    int stride1 = 1;
    MaxPooling<float, 4> pl(extents2d, stride1);

    pl.init();
    pl.forward(input);
    pl.backward(input, fakeloss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_output(), output));
    EXPECT_TRUE(is_elementwise_approx_eq(pl.get_loss_by_input_derivative(), dinput));
  }

}