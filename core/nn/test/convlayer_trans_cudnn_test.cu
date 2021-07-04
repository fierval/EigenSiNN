#include <gtest/gtest.h>
#include "layers/convolutiontrans.hpp"
#include "layers/input.hpp"
#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class TransConvolutionCudnn : public ::testing::Test {

  protected:
    void SetUp() {
      cd1p.init();
    }

    ConvDataWith1Padding<GpuDevice, RowMajor> cd1p;

    const Padding2D padding = { 1, 1 };
    const int stride = 1;
    const int dilation = 2;
  };


  TEST_F(TransConvolutionCudnn, Backward1Padding2Dilated) {

    Input<float, GpuDevice> input;
    input.set_input(cd1p.input_trans);

    TransConv2d<float, GpuDevice, RowMajor> conv2d(cd1p.kernelDims, padding, stride, dilation);
    conv2d.set_cudnn(true);

    conv2d.init(cd1p.weights.to_host());
    conv2d.forward(input);
    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.output_trans, conv2d.get_output()));

    conv2d.backward(input, cd1p.convLossTrans.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dinput_trans, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dweight_trans, conv2d.get_loss_by_weights_derivative()));
  }

}