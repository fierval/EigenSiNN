#include <iostream>
#include <gtest/gtest.h>
#include <layers/dropout.hpp>
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include "layers/input.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Dropout : public ::testing::Test {

  protected:

    void SetUp() override {
      cd.init();
    }

    CommonData2d cd;
  };

  TEST_F(Dropout, Forward) {

    Input<float, 2> input(cd.linearInput.dimensions());
    input.set_input(cd.linearInput.data());

    EigenSinn::Dropout<float, 2> dropout;
    dropout.forward(input);

    TensorMap<Tensor<float, 2>> output(dropout.get_output(), vector2array<2>(dropout.get_out_dims()));

    Tensor<float, 2> zeros(output.dimensions());
    zeros.setZero();
    
    Tensor<bool, 0> anyzeros = (output == zeros).any();
    Tensor<bool, 0> allzeros = (output == zeros).all();
    EXPECT_TRUE(anyzeros(0));
    EXPECT_FALSE(allzeros(0));

    for (Index i = 0; i < output.dimension(0); i++) {
      for (Index j = 0; j < output.dimension(1); j++) {
        EXPECT_TRUE(output(i, j) == 0. || output(i, j) == 2. * cd.linearInput(i, j));
      }
    }
  }

  TEST_F(Dropout, Backward) {

    Input<float, 2> input(cd.linearInput.dimensions());
    input.set_input(cd.linearInput.data());

    EigenSinn::Dropout<float, 2> dropout;
    dropout.forward(input);
    dropout.backward(input, cd.linearLoss.data());

    TensorMap<Tensor<float, 2>> dinput(dropout.get_loss_by_input_derivative(), vector2array<2>(dropout.get_out_dims()));

    Tensor<float, 2> zeros(dinput.dimensions());
    zeros.setZero();
    Tensor<bool, 0> anyzeros = (dinput == zeros).any();
    Tensor<bool, 0> allzeros = (dinput == zeros).all();
    EXPECT_TRUE(anyzeros(0));
    EXPECT_FALSE(allzeros(0));

    for (Index i = 0; i < dinput.dimension(0); i++) {
      for (Index j = 0; j < dinput.dimension(1); j++) {
        EXPECT_TRUE(dinput(i, j) == 0. || dinput(i, j) == 2. * cd.linearLoss(i, j));
      }
    }
  }

}