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
    res.view() = unf_kernel->contract(*col_inputs, prod_dims);

    auto conv_res = fold_conv_res(res, cd.convOutDims);
    EXPECT_TRUE(is_elementwise_approx_eq(conv_res, convolved));
  }

  TEST_F(Convolution, Backward) {

    Input<float, 4> input;
    input.set_input(cd.convInput);

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(input);
    
    auto exp_output = DeviceTensor<DefaultDevice, float, 4>(cd.output);
    EXPECT_TRUE(is_elementwise_approx_eq(exp_output, conv2d.get_output()));

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

    Tensor<float, 0> avg = weights.to_host().mean();
    
    Tensor<float, 0> var = (weights - avg(0)).to_host().pow(2.).mean();

    Tensor<float, 0> var_expected;
    var_expected.setConstant(1. / (kdims[1] * kdims[2] * kdims[3]));

    EXPECT_TRUE(is_elementwise_approx_eq(var_expected, var, 1e-4));
  }

  TEST_F(Convolution, unfold_kernel) {

    auto convWeights = DeviceTensor<DefaultDevice, float, 4>(cd.convWeights);

    auto unf_kernel = unfold_kernel(convWeights);
    auto folded_kernel = fold_kernel(unf_kernel, cd.kernelDims);

    EXPECT_TRUE(is_elementwise_approx_eq(folded_kernel.to_host(), cd.convWeights));
  }

  TEST_F(Convolution, Backward1Padding) {

    Input<float, 4> input;
    input.set_input(cd.convInput);

    Conv2d<float> conv2d(cd.kernelDims, { 1, 1 });

    conv2d.init(cd.convWeights);
    conv2d.forward(input);

    auto cd1pConvLoss = DeviceTensor<DefaultDevice, float, 4>(cd1p.convLoss);
    conv2d.backward(input, cd1pConvLoss);

    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(Convolution, FoldUnfold) {
    auto output = DeviceTensor<DefaultDevice, float, 4>(cd.output);
    auto unf_fold = fold_conv_res(unfold_conv_res(output), output.dimensions());

    EXPECT_TRUE(is_elementwise_approx_eq(output, unf_fold));
  }
}