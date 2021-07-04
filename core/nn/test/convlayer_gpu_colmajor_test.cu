#include <gtest/gtest.h>
#include "layers/convolution.hpp"
#include "layers/input.hpp"
#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class ConvolutionColMajorGpu : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
      cd1p.init();
    }

    CommonData4d<GpuDevice, ColMajor> cd;
    ConvDataWith1Padding<GpuDevice, ColMajor> cd1p;

    const Padding2D padding = { 0, 0 };
  };


  TEST_F(ConvolutionColMajorGpu, Forward) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, ColMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

  }

  TEST_F(ConvolutionColMajorGpu, Backward1Padding2Dilated) {
    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, ColMajor> conv2d(cd.kernelDims, { 1, 1 }, 1, 2);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);
    
    DeviceTensor<float, 4, GpuDevice, ColMajor> conv2dout(conv2d.get_output());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.outputDilated2Padded1, conv2dout));

    conv2d.backward(input, cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinputDilated2Padded1, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweightsDilated2Padded1, conv2d.get_loss_by_weights_derivative()));

  }

  TEST_F(ConvolutionColMajorGpu, Backward) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, ColMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);
    
    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

    conv2d.backward(input, cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(ConvolutionColMajorGpu, BackwardBias) {

    cd.init_with_bias();

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, ColMajor> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host(), cd.bias.to_host());
    conv2d.forward(input);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

    conv2d.backward(input, cd.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(ConvolutionColMajorGpu, Initialization) {

    array<Index, 4> kdims = { 1, 512, 3, 3 };
    Conv2d<float, GpuDevice, ColMajor> conv2d(kdims);

    conv2d.init();
    DeviceTensor<float, 4, GpuDevice, ColMajor> weights(conv2d.get_weights());

    Tensor<float, 0> avg = weights.to_host().mean();
    
    Tensor<float, 0> var = (weights.to_host() - avg(0)).pow(2.).mean();

    Tensor<float, 0> var_expected;
    var_expected.setConstant(1. / (kdims[1] * kdims[2] * kdims[3]));

    EXPECT_TRUE(is_elementwise_approx_eq(var_expected, var, 1e-4));
  }

  TEST_F(ConvolutionColMajorGpu, UnfoldKernel) {

    auto unf_kernel = unfold_kernel<float, GpuDevice, ColMajor>(cd.convWeights);
    auto folded_kernel = fold_kernel<float, GpuDevice, ColMajor>(unf_kernel, cd.kernelDims);

    EXPECT_TRUE(is_elementwise_approx_eq(folded_kernel, cd.convWeights));
  }

  TEST_F(ConvolutionColMajorGpu, Backward1Padding) {

    Input<float, GpuDevice> input;
    input.set_input(cd.convInput);

    Conv2d<float, GpuDevice, ColMajor> conv2d(cd.kernelDims, { 1, 1 });

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);

    conv2d.backward(input, cd1p.convLoss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(ConvolutionColMajorGpu, FoldUnfold) {
    auto output = DeviceTensor<float, 4, GpuDevice, ColMajor>(cd.output);
    auto unf_fold = fold_conv_res(unfold_conv_res(output), output.dimensions());

    EXPECT_TRUE(is_elementwise_approx_eq(output, unf_fold));
  }
}