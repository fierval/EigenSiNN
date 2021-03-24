#include <iostream>
#include <gtest/gtest.h>
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class MemoryGpu : public ::testing::Test {

  protected:
    void SetUp() {
    }
  };

  TEST_F(MemoryGpu, LargeAllocation) {

    DSizes<Index, 4> dims(10000, 3, 32, 32);

    Tensor<float, 4> large_expected(dims);
    large_expected.setRandom();

    DeviceTensor<float, 4, GpuDevice> large_gpu(large_expected);

    Tensor<float, 4> actual = large_gpu.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(large_expected, actual));

  }

}