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

    CommonData2d<ThreadPoolDevice> cd;
  };

  TEST_F(Dropout, Forward) {

    Input<float> input;
    input.set_input(cd.linearInput);

    EigenSinn::Dropout<float, 2> dropout;
    dropout.forward(input);

    auto linearInput = cd.linearInput.to_host();
    Tensor<float, 2, RowMajor> output = DeviceTensor<float, 2>(dropout.get_output()).to_host();

    Tensor<float, 2, RowMajor> zeros = DeviceTensor<float, 2>(output.dimensions()).to_host();
    zeros.setZero();
    
    Tensor<bool, 0, RowMajor> anyzeros; 
    anyzeros = (output == zeros).any();
    
    Tensor<bool, 0, RowMajor> allzeros; 
    allzeros = (output == zeros).all();

    EXPECT_TRUE(anyzeros(0));
    EXPECT_FALSE(allzeros(0));

    for (Index i = 0; i < output.dimension(0); i++) {
      for (Index j = 0; j < output.dimension(1); j++) {
        EXPECT_TRUE(output(i, j) == 0. || output(i, j) == 2. * linearInput(i, j));
      }
    }
  }

  TEST_F(Dropout, Backward) {

    Input<float> input;
    input.set_input(cd.linearInput);
    auto linearLoss = cd.linearLoss.to_host();


    EigenSinn::Dropout<float, 2> dropout;
    dropout.forward(input);
    dropout.backward(input, cd.linearLoss.raw());

    Tensor<float, 2, RowMajor> dinput = DeviceTensor<float, 2>(dropout.get_loss_by_input_derivative()).to_host();

    Tensor<float, 2, RowMajor> zeros(dinput.dimensions());
    zeros.setZero();

    Tensor<bool, 0, RowMajor> anyzeros = (dinput == zeros).any();
    Tensor<bool, 0, RowMajor> allzeros = (dinput == zeros).all();

    EXPECT_TRUE(anyzeros(0));
    EXPECT_FALSE(allzeros(0));

    for (Index i = 0; i < dinput.dimension(0); i++) {
      for (Index j = 0; j < dinput.dimension(1); j++) {
        EXPECT_TRUE(dinput(i, j) == 0. || dinput(i, j) == 2. * linearLoss(i, j));
      }
    }
  }

}