#include <iostream>
#include <gtest/gtest.h>
#include <layers/softmax.hpp>
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Softmax : public ::testing::Test {

  protected:
    void SetUp() override {

      output.resize(cd.dims);
      dinput.resize(cd.dims);

      cd.init();

      output.setValues({ {0.11150318, 0.01055052, 0.08759052, 0.06847187, 0.07035609,
          0.02311835, 0.05182620, 0.57658327},
         {0.01278833, 0.14119089, 0.01034965, 0.49899465, 0.05227858,
          0.17899387, 0.06016341, 0.04524060},
         {0.12328379, 0.05145486, 0.02898070, 0.09352909, 0.07669657,
          0.19928952, 0.14227933, 0.28448611} });

      dinput.setValues({ { 0.01045677,  0.00142033,  0.05301053,  0.02391269,  0.04195632,
           0.00558414,  0.01714944, -0.15349022},
         {-0.00065736, -0.01765234, -0.00026504, -0.05063212,  0.03444937,
           0.04745903, -0.01563458,  0.00293304},
         {-0.00967283,  0.01869225,  0.00325954,  0.00324569, -0.02049649,
           0.01421760, -0.03682784,  0.02758209} });
    }

    CommonData2d cd;
    Tensor<float, 2> output, dinput;

  };

  TEST_F(Softmax, Forward) {

    EigenSinn::Softmax<float, 2> softmax;
    softmax.init();
    softmax.forward(cd.linearInput);

    EXPECT_TRUE(is_elementwise_approx_eq(softmax.get_output(), output));
  }

  TEST_F(Softmax, Backward) {

    EigenSinn::Softmax<float, 2> softmax;
    softmax.init();
    softmax.forward(cd.linearInput);
    softmax.backward(cd.linearInput, cd.linearLoss);

    EXPECT_TRUE(is_elementwise_approx_eq(softmax.get_loss_by_input_derivative(), dinput));
  }


}