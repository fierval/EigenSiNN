
#include <gtest/gtest.h>
#include "include/commondata2d.hpp"
#include <ops/comparisons.hpp>
#include <gpu/gpu_tensor.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class CudaTest : public ::testing::Test {

  protected:

    void SetUp() override {
      cd.init();
    }

    CommonData2d cd;
  };

  TEST_F(CudaTest, Copy) {

    float * d_input = to_device(cd.linearInput);
    TensorMap<Tensor<float, 2>> gpu_input(d_input, cd.linearInput.dimensions());
    Tensor<float, 2> input = from_device(gpu_input.data(), gpu_input.dimensions());

    EXPECT_TRUE(is_elementwise_approx_eq(cd.linearInput, input.data()));
  }
}