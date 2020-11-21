#include <gtest/gtest.h>
#include "layers/linear.hpp"
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "layers/input.hpp"
#include "ops/comparisons.hpp"
#include "device/device_tensor.hpp"

using namespace Eigen;
using namespace EigenSinn;

namespace EigenSinnTest {
  class FullyConnectedGpu : public ::testing::Test {

  protected:
    void SetUp() override {
      cd.init();

    }

    CommonData2d cd;
  };

  TEST_F(FullyConnectedGpu, BackpropNoBias) {

    Input<float, 2, GpuDevice> input;
    input.set_input(cd.linearInput);

    Linear<float, ColMajor, GpuDevice> linear(cd.dims[1], cd.out_dims[1]);

    linear.init(cd.weights);
    linear.forward(input);
    linear.backward(input, DeviceTensor<GpuDevice, float, 2>(cd.fakeloss));

    EXPECT_TRUE((is_elementwise_approx_eq<float, 2, 0, GpuDevice>(cd.output, linear.get_output())));
    EXPECT_TRUE((is_elementwise_approx_eq<float, 2, 0, GpuDevice>(cd.dinput, linear.get_loss_by_input_derivative())));
    EXPECT_TRUE((is_elementwise_approx_eq<float, 2, 0, GpuDevice>(cd.dweights, linear.get_loss_by_weights_derivative())));
  }

  TEST_F(FullyConnectedGpu, BackpropBias) {

    Input<float, 2, GpuDevice> input;
    input.set_input(cd.linearInput);

    Linear<float, ColMajor, GpuDevice> linear(cd.dims[1], cd.out_dims[1]);

    linear.init(cd.weights, cd.bias);
    linear.forward(input);
    linear.backward(input, DeviceTensor<GpuDevice, float, 2>(cd.fakeloss));

    cd.output.setValues({ { 0.23947978,  1.57379603,  0.23219500,  0.99900943},
        { 1.11431766, -1.17991734, -0.41367567,  1.14438200},
        { 0.11173135,  0.14724597,  0.10992362,  1.70647931} });

    EXPECT_TRUE((is_elementwise_approx_eq<float, 2, 0, GpuDevice>(cd.output, linear.get_output())));
    EXPECT_TRUE((is_elementwise_approx_eq<float, 2, 0, GpuDevice>(cd.dinput, linear.get_loss_by_input_derivative())));
    EXPECT_TRUE((is_elementwise_approx_eq<float, 2, 0, GpuDevice>(cd.dweights, linear.get_loss_by_weights_derivative())));
    EXPECT_TRUE((is_elementwise_approx_eq<float, 1, 0, GpuDevice>(cd.dbias, linear.get_loss_by_bias_derivative())));
  }

  TEST_F(FullyConnectedGpu, Initialize) {

    int in_dim = 1024;
    Linear<float, ColMajor, GpuDevice> linear(in_dim, 512);

    linear.init();
    Tensor<float, 2> weights(DeviceTensor<GpuDevice, float, 2>((linear.get_weights())).to_host());
    Tensor<float, 0> avg = weights.mean();
    Tensor<float, 0> std = (weights - avg(0)).pow(2.).mean();

    Tensor<float, 0> std_expected;
    std_expected.setConstant(1. / in_dim);

    EXPECT_TRUE(is_elementwise_approx_eq(std_expected, std, 1e-4));
  }
}