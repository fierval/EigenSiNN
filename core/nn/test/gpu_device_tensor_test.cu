#include <device/tensor_view.hpp>
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

#include "gtest/gtest.h"

using namespace EigenSinn;

namespace EigenSinnTest {

  class DeviceTensorTestGpu : public ::testing::Test {

  protected:

    void SetUp() {
      cd.init();
    }

    CommonData4d cd;
  };

  TEST_F(DeviceTensorTestGpu, CreateEmpty) {
    DeviceTensor<GpuDevice, float, 4> d;

    EXPECT_FALSE(d);
  }

  TEST_F(DeviceTensorTestGpu, CreateDims) {
    DeviceTensor<GpuDevice, float, 4> d(4, 3, 2, 1);

    EXPECT_EQ(24, d->dimensions().TotalSize());
  }

  TEST_F(DeviceTensorTestGpu, CreateValue) {
    DeviceTensor<GpuDevice, float, 4> d(cd.convInput);

    TensorView<float, 4> h_tensor = d.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(cd.convInput, h_tensor.data()));
  }

  TEST_F(DeviceTensorTestGpu, SetFromHost) {

    DeviceTensor<GpuDevice, float, 4> d(cd.convInput);

    d.set_data_from_host(cd.convWeights);
    TensorView<float, 4> h_tensor = d.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(cd.convWeights, h_tensor.data()));

  }

}