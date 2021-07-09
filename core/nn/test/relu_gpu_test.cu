#include <iostream>
#include <gtest/gtest.h>
#include <layers/relu.hpp>
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class ReLU2dGpu : public ::testing::Test {

  protected:

    void SetUp() override {

      output.resize(cd.dims);
      dinput.resize(cd.dims);
      dinput_leaky.resize(cd.dims);
      output_leaky.resize(cd.dims);
      

      cd.init();

      output.setValues({ { 0.87322980, 0.00000000, 0.63184929, 0.38559973, 0.41274598,
          0.00000000, 0.10707247, 2.51629639} ,
        {0.00000000, 0.81226659, 0.00000000, 2.07474923, 0.00000000,
          1.04950535, 0.00000000, 0.00000000},
         {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
          0.24772950, 0.00000000, 0.60365528} });

      dinput.setValues({ { 0.36603546, 0.00000000, 0.87746394, 0.62148917, 0.86859787,
          0.00000000, 0.60315830, 0.00604892} ,
         {0.00000000, 0.1375852, 0.00000000, 0.16111487, 0.00000000,
          0.52772647, 0.00000000, 0.00000000},
         {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
          0.50212926, 0.00000000, 0.52774191} });

      dinput_leaky.setValues({ {3.66035461e-01, 4.06876858e-03, 8.77463937e-01, 6.21489167e-01,
        8.68597865e-01, 5.13801072e-03, 6.03158295e-01, 6.04892010e-03},
        {2.11180025e-03, 1.37558520e-01, 2.36974354e-03, 1.61114872e-01,
        9.21540707e-03, 5.27726471e-01, 2.71451008e-05, 3.27415229e-03},
        {3.52327945e-03, 7.94062577e-03, 5.43260481e-03, 4.65490250e-03,
        1.63546554e-03, 5.02129257e-01, 1.71945989e-03, 5.27741909e-01} });

      output_leaky.setValues({ { 8.73229802e-01, -1.48464823e-02, 6.31849289e-01, 3.85599732e-01,
        4.12745982e-01, -7.00196670e-03, 1.07072473e-01, 2.51629639e+00},
        {-1.58931315e-02, 8.12266588e-01, -1.80089306e-02, 2.07474923e+00,
        -1.81259448e-03, 1.04950535e+00, -4.07818094e-04, -3.25851166e-03},
        {-2.32540234e-03, -1.10632433e-02, -1.68039929e-02, -5.08756749e-03,
        -7.07172183e-03, 2.47729495e-01, -8.92369600e-04, 6.03655279e-01} });
    }


    CommonData2d<GpuDevice> cd;
    DeviceTensor<float, 2, GpuDevice> output, output_leaky, dinput, dinput_leaky;
    float thresh = 0.01;

  };

  TEST_F(ReLU2dGpu, Backward) {
    
    Input<float, GpuDevice> input;
    input.set_input(cd.linearInput);

    ReLU<float, 2, GpuDevice> rl;
    rl.init();
    rl.forward(input.get_output());
    rl.backward(input.get_output(), cd.linearLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(rl.get_loss_by_input_derivative(), dinput, 3e-5));
  }

  TEST_F(ReLU2dGpu, LeakyBackward) {

    Input<float, GpuDevice> input;
    input.set_input(cd.linearInput);

    LeakyReLU<float, 2, GpuDevice> rl(thresh);
    rl.init();
    rl.forward(input.get_output());
    rl.backward(input.get_output(), cd.linearLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(rl.get_loss_by_input_derivative(), dinput_leaky));
  }


}