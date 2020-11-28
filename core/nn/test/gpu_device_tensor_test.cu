#include <device/device_tensor.hpp>
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

    CommonData4d<GpuDevice> cd;
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

    Tensor<float, 4> h_tensor = d.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(cd.convInput.to_host(), h_tensor));
  }

  TEST_F(DeviceTensorTestGpu, AddTensors) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), d2(cd.convInput), sum_tensor(cd.convInput.dimensions());

    sum_tensor.view() = *d1 + *d2;

    DeviceTensor<GpuDevice, float, 4> convsum = cd.convInput + cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(convsum, sum_tensor));
  }

  TEST_F(DeviceTensorTestGpu, AddTensorsOperator) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), d2(cd.convInput), sum_tensor;

    sum_tensor = d1 + d2;

    DeviceTensor<GpuDevice, float, 4> convsum = cd.convInput + cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(convsum, sum_tensor));
  }

  TEST_F(DeviceTensorTestGpu, MultTensorsOperator) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), d2(cd.convInput), res_tensor;

    res_tensor = d1 * d2;

    DeviceTensor<GpuDevice, float, 4> expected = cd.convInput * cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, res_tensor));
  }

  TEST_F(DeviceTensorTestGpu, MultTensorConstOperator) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), res_tensor;

    res_tensor = 0.1f * d1;

    DeviceTensor<GpuDevice, float, 4> expected = 0.1 * cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, res_tensor));
  }

  TEST_F(DeviceTensorTestGpu, DivTensorsOperator) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), d2(cd.dinput), res_tensor;

    res_tensor = d1 / d2;

    DeviceTensor<GpuDevice, float, 4> expected = cd.convInput / cd.dinput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, res_tensor));
  }

  TEST_F(DeviceTensorTestGpu, SqrtTensor) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), res_tensor(d1.dimensions()), d2(cd.dinput);

    res_tensor.view() = d1->sqrt() + *d2;

    Tensor<float, 4> expected = cd.convInput.to_host().sqrt() + cd.dinput.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(expected, res_tensor.to_host()));
  }

  TEST_F(DeviceTensorTestGpu, Resize) {
    DeviceTensor<GpuDevice, float, 4> t1, t2(2, 3, 4, 5);

    t1.resize(array<Index, 4>{2, 3, 4, 5});
    t2.resize(array<Index, 4>{3, 3, 3, 3});

    EXPECT_EQ(t1.dimensions().TotalSize(), 2 * 3 * 4 * 5);
    EXPECT_EQ(t2.dimensions().TotalSize(), 3 * 3 * 3 * 3);

  }

  TEST_F(DeviceTensorTestGpu, SetConstant) {

    DeviceTensor<GpuDevice, float, 4> t1(4, 4, 4, 4);
    t1.setConstant(1);

    Tensor<float, 4> expected(4, 4, 4, 4);
    expected.setConstant(1);

    Tensor<float, 4> h_tensor = t1.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));

  }

  TEST_F(DeviceTensorTestGpu, SetZero) {

    DeviceTensor<GpuDevice, float, 4> t1(4, 4, 4, 4);
    t1.setZero();

    Tensor<float, 4> expected(4, 4, 4, 4);
    expected.setZero();

    Tensor<float, 4> h_tensor = t1.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));

  }

}