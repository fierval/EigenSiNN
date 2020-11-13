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

  TEST_F(DeviceTensorTestGpu, AddTensors) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), d2(cd.convInput), sum_tensor(cd.convInput.dimensions());

    auto device = sum_tensor.device();
    sum_tensor->device(device) = *d1 + *d2;

    TensorView<float, 4> h_tensor = sum_tensor.to_host();
    Tensor<float, 4> convsum = cd.convInput + cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(convsum, h_tensor.data()));
  }

  TEST_F(DeviceTensorTestGpu, AddTensorsOperator) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), d2(cd.convInput), sum_tensor;

    sum_tensor = d1 + d2;

    TensorView<float, 4> h_tensor = sum_tensor.to_host();
    Tensor<float, 4> convsum = cd.convInput + cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(convsum, h_tensor.data()));
  }

  TEST_F(DeviceTensorTestGpu, MultTensorsOperator) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), d2(cd.convInput), res_tensor;

    res_tensor = d1 * d2;

    TensorView<float, 4> h_tensor = res_tensor.to_host();
    Tensor<float, 4> expected = cd.convInput * cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));
  }

  TEST_F(DeviceTensorTestGpu, DivTensorsOperator) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), d2(cd.dinput), res_tensor;

    res_tensor = d1 / d2;

    TensorView<float, 4> h_tensor = res_tensor.to_host();
    Tensor<float, 4> expected = cd.convInput / cd.dinput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));
  }

  TEST_F(DeviceTensorTestGpu, SqrtTensor) {
    DeviceTensor<GpuDevice, float, 4> d1(cd.convInput), res_tensor(d1.dimensions()), d2(cd.dinput);

    auto device = res_tensor.device();
    res_tensor->device(device) = d1->sqrt() + *d2;

    TensorView<float, 4> h_tensor = res_tensor.to_host();
    Tensor<float, 4> expected = cd.convInput.sqrt() + cd.dinput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));
  }

}