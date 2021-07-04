#include <gtest/gtest.h>
#include "layers/flatten.hpp"
#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class FlatLayer : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
      cd1p.init();
    }

    CommonData4d<ThreadPoolDevice> cd;
    ConvDataWith1Padding<ThreadPoolDevice> cd1p;

    const Padding2D padding = { 0, 0 };
  };

  TEST_F(FlatLayer, Flat) {
    Input<float> input;
    input.set_input(cd.convInput.to_host());

    Flatten<float> conv2d;

    conv2d.forward(input);

    conv2d.backward(input, conv2d.get_output());


    EXPECT_TRUE(is_elementwise_approx_eq(cd.convInput, conv2d.get_loss_by_input_derivative()));
  }
}