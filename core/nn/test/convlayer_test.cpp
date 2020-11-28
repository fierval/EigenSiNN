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

    CommonData4d<DefaultDevice> cd;
    ConvDataWith1Padding<DefaultDevice> cd1p;

    const Padding2D padding = { 0, 0 };
  };

  
  TEST_F(Convolution, ForwardIm2Col) {

    Input<float, 4> input;
    input.set_input(cd.convInput);

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);
    auto convolved = conv2d.get_output();

    // perform convolutiion with GEMM using im2col
    auto col_inputs = im2col<float>(cd.convInput, cd.convWeights.dimensions(), padding);
    auto unf_kernel = unfold_kernel<float>(cd.convWeights);

    ProductDims prod_dims = { IndexPair<int>(1, 0) };
    DeviceTensor<DefaultDevice, float, 2> res(unf_kernel.dimension(0), col_inputs.dimension(1));
    res.view() = unf_kernel->contract(*col_inputs, prod_dims);

    //auto col2im_res = col2im<float>(res, convWeights.dimensions(), convInput.dimensions(), { 0, 0 });
    auto conv_res = fold_conv_res(res, cd.convOutDims);
    EXPECT_TRUE(is_elementwise_approx_eq(conv_res, convolved));
  }

  TEST_F(Convolution, Backward) {

    Input<float, 4> input;
    input.set_input(cd.convInput);

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);
    
    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));

    conv2d.backward(input, cd.convLoss);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
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

    auto unf_kernel = unfold_kernel(cd.convWeights);
    auto folded_kernel = fold_kernel(unf_kernel, cd.kernelDims);

    EXPECT_TRUE(is_elementwise_approx_eq(folded_kernel, cd.convWeights));
  }

  TEST_F(Convolution, Backward1Padding) {

    Input<float, 4> input;
    input.set_input(cd.convInput);

    Conv2d<float> conv2d(cd.kernelDims, { 1, 1 });

    conv2d.init(cd.convWeights.to_host());
    conv2d.forward(input);

    conv2d.backward(input, cd1p.convLoss);

    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dinput, conv2d.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd1p.dweight, conv2d.get_loss_by_weights_derivative()));
  }

  TEST_F(Convolution, FoldUnfold) {
    auto output = DeviceTensor<DefaultDevice, float, 4>(cd.output);
    auto unf_fold = fold_conv_res(unfold_conv_res(output), output.dimensions());

    EXPECT_TRUE(is_elementwise_approx_eq(output, unf_fold));
  }
}