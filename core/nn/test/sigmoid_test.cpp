#include <iostream>
#include <gtest/gtest.h>
#include <layers/sigmoid.hpp>
#include <layers/input.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Sigmoid : public ::testing::Test {

  protected:
    void SetUp() override {

      output.resize(cd.dims);
      dinput.resize(cd.dims);

      cd.init();

      output.setValues({{0.70541728, 0.18472636, 0.65290862, 0.59522295, 0.60174614,
          0.33176863, 0.52674258, 0.92527646},
         {0.16948053, 0.69259232, 0.14174238, 0.88842458, 0.45480880,
          0.74067992, 0.48980597, 0.41925046},
         {0.44212550, 0.24855676, 0.15704261, 0.37548503, 0.33022398,
          0.56161761, 0.47770557, 0.64649212}});

      dinput.setValues({{0.07606354, 0.06127668, 0.19884995, 0.14973700, 0.20815749,
          0.11390878, 0.15035821, 0.00041822},
         {0.02972504, 0.02928734, 0.02882828, 0.01597073, 0.22850317,
          0.10136210, 0.00067835, 0.07971890},
         {0.08690188, 0.14831208, 0.07191695, 0.10915561, 0.03617259,
          0.12362586, 0.04290103, 0.12061017}});
    }

    CommonData2d cd;
    Tensor<float, 2> output, dinput;

  };

  TEST_F(Sigmoid, Backward) {

    Input<float, 2> input(cd.dims);
    input.set_input(cd.linearInput.data());

    EigenSinn::Sigmoid<float, 2> sg;
    sg.init();
    sg.forward(input);
    sg.backward(input, cd.linearLoss.data());

    EXPECT_TRUE(is_elementwise_approx_eq(sg.get_output(), output));
    EXPECT_TRUE(is_elementwise_approx_eq(sg.get_loss_by_input_derivative(), dinput));
  }


}