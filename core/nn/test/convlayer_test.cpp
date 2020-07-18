#include <gtest/gtest.h>
#include <iostream>
#include "layers/convolution.hpp"
#include "include/commondata4d.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Convolution : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
    }

    CommonData4d cd;
  };

  TEST_F(Convolution, Forward) {
    
    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(cd.convInput);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));
  }

  TEST_F(Convolution, Backward) {

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(cd.convInput);
    conv2d.backward(cd.convInput, cd.convLoss);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  }

}