#include <iostream>
#include <gtest/gtest.h>
#include <layers/relu.hpp>
#include "include/reludata4d.hpp"
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class ReLU4dGpu : public ::testing::Test { 
  
  protected:
    void SetUp() override {
    
      rd.init();
      cd.init();
    }

    ReluData4d<GpuDevice, RowMajor> rd;
    CommonData4d<GpuDevice, RowMajor> cd;
  };

  TEST_F(ReLU4dGpu, Backward) {

    Input<float, 4, GpuDevice, RowMajor> input;
    input.set_input(rd.input);

    ReLU<float, 4, GpuDevice, RowMajor> rl;
    rl.init();
    rl.forward(input);
    rl.backward(input, cd.convInput.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(rl.get_loss_by_input_derivative(), rd.dreluInput, 3e-5));

  }

  TEST_F(ReLU4dGpu, BackwardCudnn) {

    Input<float, 4, GpuDevice, RowMajor> input;
    input.set_input(rd.input);

    ReLU<float, 4, GpuDevice, RowMajor> rl(true);
    rl.init();
    rl.forward(input);
    EXPECT_TRUE(is_elementwise_approx_eq(rl.get_output(), rd.reluOutput, 3e-5));
    
    rl.backward(input, cd.convInput.raw());

    EXPECT_TRUE(is_elementwise_approx_eq(rl.get_loss_by_input_derivative(), rd.dreluInput, 3e-5));

  }

  TEST_F(ReLU4dGpu, ForwardCudnn) {

    Input<float, 4, GpuDevice, RowMajor> input;
    input.set_input(rd.input);

    ReLU<float, 4, GpuDevice, RowMajor> rl(true);
    rl.init();
    rl.forward(input);
    EXPECT_TRUE(is_elementwise_approx_eq(rl.get_output(), rd.reluOutput, 3e-5));
  }

}
