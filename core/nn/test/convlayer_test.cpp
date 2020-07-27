#include <gtest/gtest.h>
#include <iostream>
#include "layers/convolution.hpp"
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Convolution : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
    }

    CommonData4d cd;
    const Padding2D padding = { 0, 0 };
  };

  TEST_F(Convolution, Forward) {

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(cd.convInput);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));
  }

  TEST_F(Convolution, ForwardIm2Col) {

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(cd.convInput);
    auto convolved = from_any<float, 4>(conv2d.get_output());

    // perform convolutiion with GEMM using im2col
    auto col_inputs = im2col(cd.convInput, cd.convWeights.dimensions(), padding);
    auto unf_kernel = unfold_kernel(cd.convWeights);

    ProductDims prod_dims = { IndexPair<int>(1, 0) };
    Tensor<float, 2> res = unf_kernel.contract(col_inputs, prod_dims);

    auto conv_res = fold_conv_res(res, cd.convOutDims);
    auto unf_res = unfold_conv_res(conv_res);

    EXPECT_TRUE(is_elementwise_approx_eq(conv_res, convolved));
    EXPECT_TRUE(is_elementwise_approx_eq(unf_res, res));
  }

  TEST_F(Convolution, Backward) {

    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(cd.convInput);

    Tensor<float, 2> dout_col = unfold_conv_res(cd.convLoss);

    conv2d.backward(cd.convInput, dout_col);

    // validate that we did it right by folding back to 4D
    Tensor<float, 2> dX_col = from_any<float, 2>(conv2d.get_loss_by_input_derivative());
    
    // dX folding through col2im
    Tensor<float, 4> dinput_actual = col2im(dX_col, cd.convWeights.dimensions(), cd.convInput.dimensions(), padding);

    // dW folding
    Tensor<float, 4> dkernel_actual = fold_kernel(from_any<float, 2>(conv2d.get_loss_by_weights_derivative()), cd.kernelDims);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, dinput_actual));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, dkernel_actual));
  }


  TEST_F(Convolution, prep) {

    Tensor<float, 4> chip = cd.convInput;
    array<Index, 4> starts = { 0, 0, 0, 0 };
    array<Index, 4> lengths = { 2, 3, 3, 3 };
    Tensor<float, 4> first_app = chip.slice(starts, lengths);

    Tensor<float, 4> shuffled = first_app.shuffle(array<int, 4>{ 3, 2, 1, 0 });

    TensorMap<Tensor<float, 2>> first_app_flat(shuffled.data(), 27, 2);
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
}