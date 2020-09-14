#include <iostream>
#include <gtest/gtest.h>
#include <layers/softmax.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Softmax : public ::testing::Test {

  protected:
    void SetUp() override {

      output.resize(cd.dims);
      dinput.resize(cd.dims);

      cd.init();

      output.setValues({{0.70612800, 0.08062375, 0.84257782, 0.14654712, 0.53236294,
          0.10714392, 0.37253454, 0.82919419},
         {0.06017479, 0.80167884, 0.07397472, 0.79353410, 0.29392362,
          0.61638695, 0.32133222, 0.04834229},
         {0.23369724, 0.11769743, 0.08344747, 0.05991880, 0.17371346,
          0.27646911, 0.30613321, 0.12246354}});

      dinput.setValues({{ 0.01768399,  0.02746626,  0.12683904,  0.10981361,  0.11383595,
          -0.00114783,  0.24211046, -0.13171576},
         {-0.01712979, -0.15870401, -0.08362415, -0.13601863,  0.09397244,
           0.01056348, -0.17704983,  0.02339209},
         {-0.00055420,  0.13123775, -0.04321494,  0.02620502, -0.20780841,
          -0.00941562, -0.06506063,  0.10832366}});
    }

    CommonData2d cd;
    Tensor<float, 2> output, dinput;

  };

  TEST_F(Softmax, Forward) {

    EigenSinn::Softmax<float, 2> softmax;
    softmax.init();
    softmax.forward(cd.linearInput);

    EXPECT_TRUE(is_elementwise_approx_eq(from_any<float, 2>(softmax.get_output()), output));
  }

  //TEST_F(Softmax, Backward) {

  //  EigenSinn::Softmax<float, 2> softmax;
  //  softmax.init();
  //  softmax.forward(cd.linearInput);
  //  softmax.backward(cd.linearInput, cd.linearLoss);

  //  EXPECT_TRUE(is_elementwise_approx_eq(softmax.get_loss_by_input_derivative(), dinput, 3e-5));
  //}


}