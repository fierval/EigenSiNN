#include <iostream>
#include <gtest/gtest.h>
#include <layers/relu.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class ReLU2d : public ::testing::Test {

  protected:

    void SetUp() override {

      output.resize(commonData.dims);
      dinput.resize(commonData.dims);

      commonData.init();

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
    }

    CommonData2d commonData;
    Tensor<float, 2> output, dinput;
    float thresh = 0.3;

  };

  TEST_F(ReLU2d, Forward) {

    ReLU<float, 2> rl;
    rl.init();
    rl.forward(commonData.linearInput);

    EXPECT_TRUE(is_elementwise_approx_eq(from_any<float, 2>(rl.get_output()), output));
  }

  TEST_F(ReLU2d, Backward) {

    ReLU<float, 2> rl;
    rl.init();
    rl.forward(commonData.linearInput);
    rl.backward(commonData.linearInput, commonData.linearLoss);

    EXPECT_TRUE(is_elementwise_approx_eq(from_any<float, 2>(rl.get_loss_by_input_derivative()), dinput, 3e-5));
  }


}