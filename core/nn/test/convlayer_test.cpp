#include <gtest/gtest.h>
#include <iostream>
#include "layers/convolution.hpp"
#include "include/commondata4d.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Convolution : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
    }

    CommonData4d cd;
  };

  TEST_F(Convolution, Forward) {
    
    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(cd.convInput);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));
  }

  //TEST_F(Convolution, Backward) {

  //  Conv2d<float> conv2d(cd.kernelDims);

  //  conv2d.init(cd.convWeights);
  //  conv2d.forward(cd.convInput);
  //  conv2d.backward(cd.convInput, cd.convLoss);

  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  //}


  //TEST_F(Convolution, prep) {

  //  Tensor<float, 4> chip = cd.convInput;
  //  array<Index, 4> starts = { 0, 0, 0, 0 };
  //  array<Index, 4> lengths = { 2, 3, 3, 3 };
  //  Tensor<float, 4> first_app = chip.slice(starts, lengths);
  //  
  //  Tensor<float, 4> shuffled = first_app.shuffle(array<int, 4>{ 3, 2, 1, 0 });

  //  TensorMap<Tensor<float, 2>> first_app_flat(shuffled.data(), 27, 2);

  //  std::cerr << chip << std::endl <<std::endl << "################################################" << std::endl << std::endl;
  //  std::cerr << first_app_flat << std::endl << std::endl << "################################################" << std::endl << std::endl;

  //}

  TEST_F(Convolution, im2col) {

    auto output = im2col(cd.convInput, cd.convWeights.dimensions(), { 0, 0 });
    EXPECT_TRUE(is_elementwise_approx_eq(output, cd.convInputUnrolledPad0Stride1));
  }

  TEST_F(Convolution, col2im) {

    auto output = im2col(cd.convInput, cd.convWeights.dimensions(), { 0, 0 });
    auto input = col2im(output, cd.convWeights.dimensions(), cd.convInput.dimensions(), { 0, 0 });

  std::cerr << input << std::endl <<std::endl << "################################################" << std::endl << std::endl;
  std::cerr << cd.convInput << std::endl << std::endl << "################################################" << std::endl << std::endl;

    EXPECT_TRUE(is_elementwise_approx_eq(input, cd.convInput));
  }
}