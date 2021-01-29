#include <gtest/gtest.h>
#include "layers/convolutiontrans.hpp"
#include "layers/input.hpp"
#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class TransConvolution : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
      cd1p.init();
    }

    CommonData4d<DefaultDevice> cd;
    ConvDataWith1Padding<DefaultDevice> cd1p;

    const Padding2D padding = { 0, 0 };
  };

  
  TEST_F(TransConvolution, Forward) {

    Input<float, 4> input;
    input.set_input(cd1p.input_trans);

    TransConv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);
    
    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

    //conv2d.backward(input, cd.convLoss);

    //EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    //EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  }


  //TEST_F(TransConvolution, Backward1Padding2Dilated) {
  //  Input<float, 4> input;
  //  input.set_input(cd.convInput);

  //  Conv2d<float> conv2d(cd.kernelDims, { 1, 1 }, 1, 2);

  //  conv2d.init(cd.convWeights.to_host());
  //  conv2d.forward(input);
  //  DeviceTensor<DefaultDevice, float, 4> conv2dout(conv2d.get_output());
  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.outputDilated2Padded1, conv2dout));

  //  conv2d.backward(input, cd.convLoss);

  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dinputDilated2Padded1, conv2d.get_loss_by_input_derivative()));
  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dweightsDilated2Padded1, conv2d.get_loss_by_weights_derivative()));

  //}

  //TEST_F(TransConvolution, Forward1Padding2Dilated) {
  //  Input<float, 4> input;
  //  input.set_input(cd.convInput);

  //  Conv2d<float> conv2d(cd.kernelDims, { 1, 1 }, 1, 2);

  //  conv2d.init(cd.convWeights.to_host());
  //  conv2d.forward(input);
  //  DeviceTensor<DefaultDevice, float, 4> conv2dout(conv2d.get_output());
  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.outputDilated2Padded1, conv2dout));
  //}

}