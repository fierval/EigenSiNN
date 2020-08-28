#include <gtest/gtest.h>
#include "layers/flatten.hpp"
#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class FlatLayer : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
      cd1p.init();
    }

    CommonData4d cd;
    ConvDataWith1Padding cd1p;

    const Padding2D padding = { 0, 0 };
  };

  TEST_F(FlatLayer, Flat) {

    Flatten<float> conv2d;

    conv2d.forward(cd.convInput);

    auto post_flat = conv2d.get_output();
    conv2d.backward(cd.convInput, post_flat);


    EXPECT_TRUE(is_elementwise_approx_eq(cd.convInput, conv2d.get_loss_by_input_derivative()));
  }
}