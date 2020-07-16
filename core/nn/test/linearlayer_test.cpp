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
      weights.setValues({ { 0.30841491,  0.19089887, -0.15496132, -0.28125578},
        { 0.16301581, -0.24521475,  0.15089461, -0.15781732},
        { 0.05912393,  0.27066174,  0.16939566, -0.32488498},
        { 0.32572004, -0.00526837, -0.25025240, -0.08520141},
        {-0.00591815, -0.18401390, -0.18078347, -0.27685770},
        {-0.07333553, -0.20650741, -0.07853529, -0.02988693},
        { 0.16375038, -0.28048125, -0.32877934,  0.18739149},
        {-0.35274175,  0.29642352,  0.19627282,  0.32216403} });

      dweights.resize(weights.dimensions());
      dweights.setValues({ {-0.06577596, -0.76513994,  0.19567230, -0.94930124},
        {-0.69478256, -0.91280013, -1.64919329,  0.16104239},
        {-0.80080771, -2.13387632, -0.83714628, -1.21797860},
        {-0.09220973,  0.76717538,  0.64137232,  1.63494349},
        {-0.28961062, -0.55723274, -0.06491914, -0.04288082},
        { 0.07141823,  0.54735136, -0.18686420,  0.59569913},
        {-0.02981755, -0.06387962,  0.03463056, -0.00357590},
        { 0.61958170,  1.04655457,  2.41507840,  0.47002986} });

      bias.resize(array<Index, 1>({ outDims[1] }));
      bias.setValues({ 0.87039185, 0.08955163, 0.14195210, 0.51105964 });
      
      dbias.resize(array<Index, 1>({ outDims[1] }));
      dbias.setValues({ 0.66265798, 1.62600470, 1.67406428, 1.05092955 });
    }

    CommonData2d commonData;
    Tensor<float, 2> output, dinput, fakeloss, weights, dweights;
    Tensor<float, 1> bias, dbias;
    array<Index, 2> outDims = { 3, 4 };
  };

  TEST_F(FullyConnected, BackpropNoBias) {

    Linear<float> linear(commonData.dims[0], commonData.dims[1], outDims[1], false);

    linear.init(weights);
    linear.forward(commonData.linearInput);
    linear.backward(commonData.linearInput, fakeloss);

    EXPECT_TRUE(is_elementwise_approx_eq(output, linear.get_output()));
    EXPECT_TRUE(is_elementwise_approx_eq(dinput, linear.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(dweights, linear.get_loss_by_weights_derivative()));
  }

  TEST_F(FullyConnected, BackpropBias) {

    Linear<float> linear(commonData.dims[0], commonData.dims[1], outDims[1], true);

    linear.init(weights, bias);
    linear.forward(commonData.linearInput);
    linear.backward(commonData.linearInput, fakeloss);
    
    output.setValues({ { 0.23947978,  1.57379603,  0.23219500,  0.99900943},
        { 1.11431766, -1.17991734, -0.41367567,  1.14438200},
        { 0.11173135,  0.14724597,  0.10992362,  1.70647931} });

    EXPECT_TRUE(is_elementwise_approx_eq(output, linear.get_output()));
    EXPECT_TRUE(is_elementwise_approx_eq(dinput, linear.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(dweights, linear.get_loss_by_weights_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(dbias, linear.get_loss_by_bias_derivative()));
  }

}