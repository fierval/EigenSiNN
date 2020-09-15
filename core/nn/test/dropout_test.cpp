#include <iostream>
#include <gtest/gtest.h>
#include <layers/dropout.hpp>
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Dropout : public ::testing::Test {

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
    }

    CommonData2d cd;
    Tensor<float, 2> output, output_leaky, dinput, dinput_leaky;
    float thresh = 0.01;

  };

  TEST_F(Dropout, Forward) {

    EigenSinn::Dropout<float, 2> dropout;
    dropout.forward(cd.linearInput);

    EXPECT_TRUE(is_elementwise_approx_eq(dropout.get_output(), output));
  }

  TEST_F(Dropout, Backward) {

    EigenSinn::Dropout<float, 2> dropout;
    dropout.forward(cd.linearInput);
    dropout.backward(cd.linearInput, cd.linearLoss);

    EXPECT_TRUE(is_elementwise_approx_eq(dropout.get_loss_by_input_derivative(), dinput, 3e-5));
  }
}