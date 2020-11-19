#include <device/tensor_view.hpp>
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

#include "gtest/gtest.h"

using namespace EigenSinn;

namespace EigenSinnTest {

  class DeviceTensorTest : public ::testing::Test {

  protected:
  
    void SetUp() {
      cd.init();
    }

    CommonData4d cd;
  };

  TEST_F(DeviceTensorTest, CreateEmpty) {
    DeviceTensor<ThreadPoolDevice, float, 4> d;

    EXPECT_FALSE(d);
  }

  TEST_F(DeviceTensorTest, CreateDims) {
    DeviceTensor<ThreadPoolDevice, float, 4> d(4, 3, 2, 1);

    EXPECT_EQ(24, d->dimensions().TotalSize());
  }

  TEST_F(DeviceTensorTest, CreateValue) {
    DeviceTensor<ThreadPoolDevice, float, 4> d(cd.convInput);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.convInput, d->data()));
  }

  TEST_F(DeviceTensorTest, SetFromHost) {

    DeviceTensor<ThreadPoolDevice, float, 4> d(cd.convInput);

    d.set_from_host(cd.convWeights);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.convWeights, d->data()));

  }

  TEST_F(DeviceTensorTest, AddTensorsOperator) {
    DeviceTensor<ThreadPoolDevice, float, 4> d1(cd.convInput), d2(cd.convInput), sum_tensor;

    sum_tensor = d1 + d2;

    Tensor<float, 4> h_tensor = sum_tensor.to_host();
    Tensor<float, 4> convsum = cd.convInput + cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(convsum, h_tensor));
  }

  TEST_F(DeviceTensorTest, MultTensorsOperator) {
    DeviceTensor<ThreadPoolDevice, float, 4> d1(cd.convInput), d2(cd.convInput), res_tensor;

    res_tensor = d1 * d2;

    Tensor<float, 4> h_tensor = res_tensor.to_host();
    Tensor<float, 4> expected = cd.convInput * cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));
  }

  TEST_F(DeviceTensorTest, DivTensorsOperator) {
    DeviceTensor<ThreadPoolDevice, float, 4> d1(cd.convInput), d2(cd.dinput), res_tensor;

    res_tensor = d1 / d2;

    Tensor<float, 4> h_tensor = res_tensor.to_host();
    Tensor<float, 4> expected = cd.convInput / cd.dinput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));
  }

  TEST_F(DeviceTensorTest, ThreeTensorsOperator) {
    DeviceTensor<ThreadPoolDevice, float, 4> d1(cd.convInput), d2(cd.dinput), d3(cd.convInput), res_tensor;

    res_tensor = d1 / d2 + d3;

    Tensor<float, 4> h_tensor = res_tensor.to_host();
    Tensor<float, 4> expected = cd.convInput / cd.dinput + cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));
  }


  TEST_F(DeviceTensorTest, SqrtTensor) {
    DeviceTensor<ThreadPoolDevice, float, 4> d1(cd.convInput), res_tensor(d1.dimensions()), d2(cd.dinput);

    res_tensor.view() = d1->sqrt() + *d2;

    Tensor<float, 4> h_tensor = res_tensor.to_host();
    Tensor<float, 4> expected = cd.convInput.sqrt() + cd.dinput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor.data()));
  }

  TEST_F(DeviceTensorTest, Resize) {
    DeviceTensor<ThreadPoolDevice, float, 4> t1, t2(2, 3, 4, 5);

    t1.resize(array<Index, 4>{2, 3, 4, 5});
    t2.resize(array<Index, 4>{3, 3, 3, 3});

    EXPECT_EQ(t1.dimensions().TotalSize(), 2 * 3 * 4 * 5);
    EXPECT_EQ(t2.dimensions().TotalSize(), 3 * 3 * 3 * 3);

  }

  TEST_F(DeviceTensorTest, SetConstant) {

    DeviceTensor<ThreadPoolDevice, float, 4> t1(4, 4, 4, 4);
    t1.setConstant(1);

    Tensor<float, 4> expected(4, 4, 4, 4);
    expected.setConstant(1);

    EXPECT_TRUE(is_elementwise_approx_eq(expected, t1->data()));

  }

  TEST_F(DeviceTensorTest, SetZero) {

    DeviceTensor<ThreadPoolDevice, float, 4> t1(4, 4, 4, 4);
    t1.setZero();

    Tensor<float, 4> expected(4, 4, 4, 4);
    expected.setZero();

    Tensor<float, 4> h_tensor = t1.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor));
  }

}