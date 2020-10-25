#include <gtest/gtest.h>
#include "layers/linear.hpp"
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "layers/input.hpp"
#include "ops/comparisons.hpp"
#include "gpu/gpu_tensor.hpp"

using namespace Eigen;
using namespace EigenSinn;

namespace EigenSinnTest {
  class LinearGpu : public ::testing::Test {

  protected:
    void SetUp() override {
      cd.init();

    }

    CommonData2d cd;
  };

  //TEST_F(LinearGpu, Backward) {
  //  Dispatcher<GpuDevice> device;
  //  float* d_input = to_device(cd.linearInput);

  //  Input<float, 2, GpuDevice> input(cd.dims, device);
  //  input.set_input(d_input);

  //  Linear<float, GpuDevice> linear(cd.dims[1], cd.out_dims[1]);

  //  linear.init(cd.weights, cd.bias);
  //  linear.forward(input);
  //  linear.backward(input, cd.fakeloss.data());

  //  cd.output.setValues({ { 0.23947978,  1.57379603,  0.23219500,  0.99900943},
  //      { 1.11431766, -1.17991734, -0.41367567,  1.14438200},
  //      { 0.11173135,  0.14724597,  0.10992362,  1.70647931} });

  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.output, linear.get_output()));
  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, linear.get_loss_by_input_derivative()));
  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dweights, linear.get_loss_by_weights_derivative()));
  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dbias, linear.get_loss_by_bias_derivative()));

  //}
}