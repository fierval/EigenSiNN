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

    d.set_data_from_host(cd.convWeights);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.convWeights, d->data()));

  }

}