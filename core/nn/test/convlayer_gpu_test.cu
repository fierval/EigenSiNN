#include <gtest/gtest.h>
#include "layers/convolution.hpp"
#include "layers/input.hpp"
#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class ConvolutionRowMajorGpu : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
      cd1p.init();
    }

    CommonData4d<GpuDevice, RowMajor> cd;
    ConvDataWith1Padding<GpuDevice, RowMajor> cd1p;

    const Padding2D padding = { 0, 0 };
  };


  TEST_F(ConvolutionRowMajorGpu, Forward) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, RowMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input.get_output());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

  }

  TEST_F(ConvolutionRowMajorGpu, Backward1Padding2Dilated) {
    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, RowMajor> conv2d(cd.kernelDims, { 1, 1 }, 1, 2);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input.get_output());
    
    DeviceTensor<float, 4, GpuDevice, RowMajor> conv2dout(conv2d.get_output());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.outputDilated2Padded1, conv2dout));

    conv2d.backward(input.get_output(), cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinputDilated2Padded1, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweightsDilated2Padded1, conv2d.get_loss_by_weights_derivative()));

  }

  TEST_F(ConvolutionRowMajorGpu, Backward) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, RowMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input.get_output());
    
    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

    conv2d.backward(input.get_output(), cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(ConvolutionRowMajorGpu, BackwardBias) {

    cd.init_with_bias();

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, RowMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host(), cd.bias.to_host());
    conv2d.forward(input.get_output());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

    conv2d.backward(input.get_output(), cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(ConvolutionRowMajorGpu, Initialization) {

    array<Index, 4> kdims = { 1, 512, 3, 3 };
    Conv2d<float, GpuDevice, RowMajor> conv2d(kdims);

    conv2d.init();
    DeviceTensor<float, 4, GpuDevice> weights(conv2d.get_weights());

    Tensor<float, 0, RowMajor> avg = weights.to_host().mean();
    
    Tensor<float, 0, RowMajor> var = (weights.to_host() - avg(0)).pow(2.).mean();

    Tensor<float, 0, RowMajor> var_expected;
    var_expected.setConstant(1. / (kdims[1] * kdims[2] * kdims[3]));

    EXPECT_TRUE(is_elementwise_approx_eq(var_expected, var, 1e-4));
  }

  TEST_F(ConvolutionRowMajorGpu, UnfoldKernel) {

    auto unf_kernel = unfold_kernel<float, GpuDevice>(cd.convWeights);
    auto folded_kernel = fold_kernel<float, GpuDevice>(unf_kernel, cd.kernelDims);

    EXPECT_TRUE(is_elementwise_approx_eq(folded_kernel, cd.convWeights));
  }

  TEST_F(ConvolutionRowMajorGpu, Backward1Padding) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, RowMajor> conv2d(cd.kernelDims, { 1, 1 });

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input.get_output());

    conv2d.backward(input.get_output(), cd1p.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(ConvolutionRowMajorGpu, FoldUnfold) {
    auto output = DeviceTensor<float, 4, GpuDevice, RowMajor>(cd.output);
    auto unf_fold = fold_conv_res(unfold_conv_res(output), output.dimensions());

    EXPECT_TRUE(is_elementwise_approx_eq(output, unf_fold));
  }
}