#include <iostream>
#include <gtest/gtest.h>
#include <layers/sigmoid.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Sigmoid : public ::testing::Test {

  protected:
    void SetUp() override {
      output.resize(cd.dims);
      dinput.resize(cd.dims);
      dinput_sigmoid.resize(cd.dims);
      output_sigmoid.resize(cd.dims);


      cd.init();

    }

    CommonData2d cd;
    Tensor<float, 2> output, output_sigmoid, dinput, dinput_sigmoid;

  };

  TEST_F(Sigmoid, Forward) {

    EigenSinn::Sigmoid<float, 2> rl;
    rl.init();
    rl.forward(cd.linearInput);

    EXPECT_TRUE(is_elementwise_approx_eq(from_any<float, 2>(rl.get_output()), output));
  }

  TEST_F(Sigmoid, Backward) {

    EigenSinn::Sigmoid<float, 2> rl;
    rl.init();
    rl.forward(cd.linearInput);
    rl.backward(cd.linearInput, cd.linearLoss);

    EXPECT_TRUE(is_elementwise_approx_eq(from_any<float, 2>(rl.get_loss_by_input_derivative()), dinput, 3e-5));
  }


}