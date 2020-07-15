#include "layers/linear.hpp"
#include "Random.h"
#include "include/commondata2d.hpp"
#include <iostream>
#include "ops/conversions.hpp"
#include "ops/comparisons.hpp"

#include <gtest/gtest.h>


using namespace EigenSinn;

namespace EigenSinnTest {

  class FullyConnected : public ::testing::Test {
  protected:
    void SetUp() override {

      commonData.init();
      dinput.resize(commonData.dims);

      output.resize(outDims);
      output.setValues({ {-0.63091207,  1.48424447,  0.09024291,  0.48794979},
        { 0.24392587, -1.26946902, -0.55562776,  0.63332230},
        {-0.75866050,  0.05769435, -0.03202848,  1.19541967} });

      dinput.resize(commonData.dims);
      dinput.setValues({ {-0.11519448,  0.03996181,  0.14310896, -0.19761968, -0.28686517,
          -0.14561312, -0.29164833,  0.29556170},
         {-0.13882221, -0.19504318, -0.05183563, -0.12013839, -0.35180071,
          -0.15447669, -0.08925162,  0.43116185},
         { 0.21379752, -0.04886286,  0.33015090,  0.01655336, -0.25806439,
          -0.24717081, -0.32011586,  0.18866043} });

      fakeloss.resize(outDims);
      fakeloss.setValues({ {0.13770211, 0.28582627, 0.86899745, 0.27578735},
        {0.04713255, 0.51820499, 0.27709258, 0.74432141},
        {0.47782332, 0.82197350, 0.52797425, 0.03082085} });

      weights.resize(commonData.dims[1], outDims[1]);
      weights.setValues({{ 0.30841491,  0.19089887, -0.15496132, -0.28125578},
        { 0.16301581, -0.24521475,  0.15089461, -0.15781732},
        { 0.05912393,  0.27066174,  0.16939566, -0.32488498},
        { 0.32572004, -0.00526837, -0.25025240, -0.08520141},
        {-0.00591815, -0.18401390, -0.18078347, -0.27685770},
        {-0.07333553, -0.20650741, -0.07853529, -0.02988693},
        { 0.16375038, -0.28048125, -0.32877934,  0.18739149},
        {-0.35274175,  0.29642352,  0.19627282,  0.32216403}});
      
    }

    CommonData2d commonData;
    Tensor<float, 2> output, dinput, fakeloss, weights;
    array<Index, 2> outDims = { 3, 4 };
  };

  TEST_F(FullyConnected, Backprop) {

    Linear<float> linear(commonData.dims[0], commonData.dims[1], outDims[1], false);

    linear.init(weights);
    linear.forward(commonData.linearInput);
    linear.backward(commonData.linearInput, commonData.linearLoss);
    
    EXPECT_TRUE(is_elementwise_approx_eq(output, from_any<float, 2>(linear.get_output())));
    EXPECT_TRUE(is_elementwise_approx_eq(dinput, from_any<float, 2>(linear.get_loss_by_input_derivative())));
  }
}