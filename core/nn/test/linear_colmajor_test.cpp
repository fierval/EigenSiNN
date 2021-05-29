#include "layers/linear.hpp"
#include "layers/input.hpp"
#include "include/commondata2d.hpp"
#include <iostream>
#include "ops/comparisons.hpp"
#include "include/testutils.hpp"

#include <gtest/gtest.h>


using namespace EigenSinn;

namespace EigenSinnTest {

  class FullyConnectedColMajor : public ::testing::Test {
  protected:
    void SetUp() override {

      cd.init();
    }
    CommonData2d<ThreadPoolDevice, ColMajor> cd;

  };

  TEST_F(FullyConnectedColMajor, BackpropNoBias) {

    Input<float, 2, ThreadPoolDevice, ColMajor> input;
    input.set_input(cd.linearInput);

    Linear<float, ThreadPoolDevice, ColMajor> linear(cd.dims[1], cd.out_dims[1]);

    linear.init(cd.weights.to_host());
    linear.forward(input);
    linear.backward(input, cd.fakeloss.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, linear.get_output()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, linear.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweights, linear.get_loss_by_weights_derivative()));
  }

  TEST_F(FullyConnectedColMajor, BackpropBias) {

    Input<float, 2, ThreadPoolDevice, ColMajor> input;
    input.set_input(cd.linearInput);

    Linear<float, ThreadPoolDevice, ColMajor> linear(cd.dims[1], cd.out_dims[1]);

    linear.init(cd.weights.to_host(), cd.bias.to_host());
    linear.forward(input);
    linear.backward(input, cd.fakeloss.raw());

    cd.output.setValues({ { 0.23947978,  1.57379603,  0.23219500,  0.99900943},
        { 1.11431766, -1.17991734, -0.41367567,  1.14438200},
        { 0.11173135,  0.14724597,  0.10992362,  1.70647931} });

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, linear.get_output()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, linear.get_loss_by_input_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweights, linear.get_loss_by_weights_derivative()));
    EXPECT_TRUE(is_elementwise_approx_eq(cd.dbias, linear.get_loss_by_bias_derivative()));
  }

  TEST_F(FullyConnectedColMajor, Initialize) {

    int in_dim = 1024;
    Linear<float, ThreadPoolDevice, ColMajor> linear(in_dim, 512);

    linear.init();
    Tensor<float, 2, ColMajor> weights(DeviceTensor<float, 2, ThreadPoolDevice, ColMajor>((linear.get_weights())).to_host());
    Tensor<float, 0, ColMajor> avg = weights.mean();
    Tensor<float, 0, ColMajor> std = (weights - avg(0)).pow(2.).mean();

    Tensor<float, 0, ColMajor> std_expected;
    std_expected.setConstant(1. / in_dim);

    EXPECT_TRUE(is_elementwise_approx_eq(std_expected, std, 1e-4));
  }

}