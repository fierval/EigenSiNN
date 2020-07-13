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
      fakeloss.resize(commonData.dims);
      dinput.resize(commonData.dims);

      commonData.init();

      output.setValues({ { 0.87322980, 0.63184929, 2.51629639},
        {2.07474923, 2.07474923, 1.04950535},
        {-0.23254025, 0.24772950, 0.60365528} });

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
    rl.backward(commonData.linearInput, fakeloss);

    EXPECT_TRUE(is_elementwise_approx_eq(from_any<float, 2>(rl.get_loss_by_input_derivative()), dinput));
  }


}