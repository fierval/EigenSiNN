#include <device/device_tensor.hpp>
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

    CommonData4d<ThreadPoolDevice> cd;
  };

  TEST_F(DeviceTensorTest, CreateEmpty) {
    DeviceTensor<float, 4> d;

    EXPECT_FALSE(d);
  }

  TEST_F(DeviceTensorTest, CreateDims) {
    DeviceTensor<float, 4> d(4, 3, 2, 1);

    EXPECT_EQ(24, d->dimensions().TotalSize());
  }

  TEST_F(DeviceTensorTest, CreateValue) {
    DeviceTensor<float, 4> d(cd.convInput);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.convInput, d));
  }

  TEST_F(DeviceTensorTest, AddTensors) {
    DeviceTensor<float, 4> d1(cd.convInput), d2(cd.convInput), sum_tensor(cd.convInput.dimensions());

    sum_tensor.view() = *d1 + *d2;

    Tensor<float, 4, RowMajor> convsum  = *cd.convInput + *cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(convsum, sum_tensor));
  }

  TEST_F(DeviceTensorTest, MultTensors) {
    DeviceTensor<float, 4> d1(cd.convInput), d2(cd.convInput), res_tensor(cd.convInput.dimensions());

    res_tensor.view() = *d1 * *d2;

    Tensor<float, 4, RowMajor> expected =*cd.convInput * *cd.convInput;

    EXPECT_TRUE(is_elementwise_approx_eq(expected, res_tensor));
  }

  TEST_F(DeviceTensorTest, SqrtTensor) {
    DeviceTensor<float, 4> d1(cd.convInput), res_tensor(d1.dimensions()), d2(cd.dinput);

    res_tensor.view() = d1->sqrt() + *d2;

    Tensor<float, 4, RowMajor> h_tensor = res_tensor.to_host();
    Tensor<float, 4, RowMajor> expected = cd.convInput.to_host().sqrt() + cd.dinput.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor));
  }

  TEST_F(DeviceTensorTest, Resize) {
    DeviceTensor<float, 4> t1, t2(2, 3, 4, 5);

    t1.resize(array<Index, 4>{2, 3, 4, 5});
    t2.resize(array<Index, 4>{3, 3, 3, 3});

    EXPECT_EQ(t1.dimensions().TotalSize(), 2 * 3 * 4 * 5);
    EXPECT_EQ(t2.dimensions().TotalSize(), 3 * 3 * 3 * 3);

  }

  TEST_F(DeviceTensorTest, SetConstant) {

    DeviceTensor<float, 4> t1(4, 4, 4, 4);
    t1.setConstant(1);

    Tensor<float, 4, RowMajor> expected(4, 4, 4, 4);
    expected.setConstant(1);

    EXPECT_TRUE(is_elementwise_approx_eq(expected, t1->data()));

  }

  TEST_F(DeviceTensorTest, SetZero) {

    DeviceTensor<float, 4> t1(4, 4, 4, 4);
    t1.setZero();

    Tensor<float, 4, RowMajor> expected(4, 4, 4, 4);
    expected.setZero();

    Tensor<float, 4, RowMajor> h_tensor = t1.to_host();

    EXPECT_TRUE(is_elementwise_approx_eq(expected, h_tensor));
  }

  TEST_F(DeviceTensorTest, CopyTensor) {

    DeviceTensor<float, 4> cp_input(cd.convInput);

    EXPECT_EQ((*cp_input)(0, 0, 0, 0), (*cd.convInput)(0, 0, 0, 0));
  }

  TEST_F(DeviceTensorTest, MoveTensor) {

    DeviceTensor<float, 4> cp_input(cd.convInput);
    DeviceTensor<float, 4> mv_input(std::move(cp_input));

    EXPECT_EQ((*cp_input)(0, 0, 0, 0), (*mv_input)(0, 0, 0, 0));
  }

}