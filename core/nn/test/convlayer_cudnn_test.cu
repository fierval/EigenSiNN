#include <gtest/gtest.h>
#include "layers/convolution_cudnn.hpp"
#include "layers/input.hpp"
#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class ConvolutionCudnn : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
      cd1p.init();
    }

    CommonData4d<CudnnDevice, RowMajor> cd;
    ConvDataWith1Padding<CudnnDevice, RowMajor> cd1p;

    const Padding2D padding = { 0, 0 };
  };


  TEST_F(ConvolutionCudnn, Forward) {

    Input<float, 4, CudnnDevice, RowMajor> input;
    input.set_input(cd.convInput);

    Conv2d<float, CudnnDevice, RowMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

  }

  TEST_F(ConvolutionCudnn, Backward1Padding2Dilated) {
    Input<float, 4, CudnnDevice, RowMajor> input;
    input.set_input(cd.convInput);

    Conv2d<float, CudnnDevice, RowMajor> conv2d(cd.kernelDims, { 1, 1 }, 1, 2);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);
    
    DeviceTensor<float, 4, CudnnDevice, RowMajor> conv2dout(conv2d.get_output());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.outputDilated2Padded1, conv2dout));

    conv2d.backward(input, cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinputDilated2Padded1, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweightsDilated2Padded1, conv2d.get_loss_by_weights_derivative()));

  }

  TEST_F(ConvolutionCudnn, Backward) {

    Input<float, 4, CudnnDevice, RowMajor> input;
    input.set_input(cd.convInput);

    Conv2d<float, CudnnDevice, RowMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);
    
    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

    conv2d.backward(input, cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(ConvolutionCudnn, BackwardBias) {

    cd.init_with_bias();

    Input<float, 4, CudnnDevice, RowMajor> input;
    input.set_input(cd.convInput);

    Conv2d<float, CudnnDevice, RowMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host(), cd.bias.to_host());
    conv2d.forward(input);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

    conv2d.backward(input, cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(ConvolutionCudnn, Initialization) {

    array<Index, 4> kdims = { 1, 512, 3, 3 };
    Conv2d<float, CudnnDevice, RowMajor> conv2d(kdims);

    conv2d.init();
    DeviceTensor<float, 4, CudnnDevice, RowMajor> weights(conv2d.get_weights());

    Tensor<float, 0, RowMajor> avg = weights.to_host().mean();
    
    Tensor<float, 0, RowMajor> var = (weights.to_host() - avg(0)).pow(2.).mean();

    Tensor<float, 0, RowMajor> var_expected;
    var_expected.setConstant(1. / (kdims[1] * kdims[2] * kdims[3]));

    EXPECT_TRUE(is_elementwise_approx_eq(var_expected, var, 1e-4));
  }

  TEST_F(ConvolutionCudnn, Backward1Padding) {

    Input<float, 4, CudnnDevice, RowMajor> input;
    input.set_input(cd.convInput);

    Conv2d<float, CudnnDevice, RowMajor> conv2d(cd.kernelDims, { 1, 1 });

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);

    conv2d.backward(input, cd1p.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dweight, conv2d.get_loss_by_weights_derivative()));
  }

}