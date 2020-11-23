#include <gtest/gtest.h>
#include "layers/convolution.hpp"
#include "layers/input.hpp"
#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Convolution : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
      cd1p.init();
    }

    CommonData4d cd;
    ConvDataWith1Padding cd1p;

    const Padding2D padding = { 0, 0 };
  };

  
  TEST_F(Convolution, ForwardIm2Col) {

    Input<float, 4> input;
    input.set_input(cd.convInput);

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(input);
    auto convolved = conv2d.get_output();
    auto convInput = DeviceTensor<DefaultDevice, float, 4>(cd.convInput);
    auto convWeights = DeviceTensor<DefaultDevice, float, 4>(cd.convWeights);

    // perform convolutiion with GEMM using im2col
    auto col_inputs = im2col<float>(convInput, convWeights.dimensions(), padding);
    auto unf_kernel = unfold_kernel<float>(convWeights);

    ProductDims prod_dims = { IndexPair<int>(1, 0) };
    DeviceTensor<DefaultDevice, float, 2> res(unf_kernel.dimension(0), col_inputs.dimension(1));
    res.view() = unf_kernel->contract(col_inputs, prod_dims);

    auto conv_res = fold_conv_res(res, cd.convOutDims);
    auto unf_res = unfold_conv_res(conv_res);

    EXPECT_TRUE(is_elementwise_approx_eq(conv_res, convolved));
    EXPECT_TRUE(is_elementwise_approx_eq(unf_res, res));
  }

  TEST_F(Convolution, Backward) {

    Input<float, 4> input;
    input.set_input(cd.convInput);

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(input);
    conv2d.backward(input, DeviceTensor<DefaultDevice, float, 4>(cd.convLoss));

    auto exp_dinput = DeviceTensor<DefaultDevice, float, 4>(cd.dinput);
    auto exp_dweight = DeviceTensor<DefaultDevice, float, 4>(cd.dweight);

    EXPECT_TRUE(is_elementwise_approx_eq(exp_dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(exp_dweight, conv2d.get_loss_by_weights_derivative()));
  }


  TEST_F(Convolution, Initialization) {

    array<Index, 4> kdims = { 1, 512, 3, 3 };
    Conv2d<float> conv2d(kdims);

    conv2d.init();
    DeviceTensor<DefaultDevice, float, 4> weights(conv2d.get_weights());

    DeviceTensor<DefaultDevice, float, 0> avg;
    avg.view() = weights->mean();

    DeviceTensor<DefaultDevice, float, 0> var;
    var.view() = (weights - avg->operator()(0))->pow(2.).mean();

    Tensor<float, 0> var_expected;
    var_expected.setConstant(1. / (kdims[1] * kdims[2] * kdims[3]));

    EXPECT_TRUE(is_elementwise_approx_eq(var_expected, var.to_host(), 1e-4));
  }

  TEST_F(Convolution, im2col) {

    auto output = im2col(cd.convInput, cd.convWeights.dimensions(), { 0, 0 });
    EXPECT_TRUE(is_elementwise_approx_eq(output, cd.convInputUnrolledPad0Stride1));
  }

  TEST_F(Convolution, unfold_kernel) {

    auto unf_kernel = unfold_kernel(cd.convWeights);
    auto folded_kernel = fold_kernel(unf_kernel, cd.kernelDims);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.convWeightsFlat, unf_kernel));
    EXPECT_TRUE(is_elementwise_approx_eq(folded_kernel, cd.convWeights));
  }

  TEST_F(Convolution, Backward1Padding) {

    Input<float, 4> input(cd.convInput.dimensions());
    input.set_input(cd.convInput.data());

    Conv2d<float> conv2d(cd.kernelDims, { 1, 1 });

    conv2d.init(cd.convWeights);
    conv2d.forward(input);

    conv2d.backward(input, cd1p.convLoss.data());

    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(Convolution, FoldUnfold) {
    auto unf_fold = fold_conv_res(unfold_conv_res(cd.output), cd.output.dimensions());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, unf_fold));
  }
}