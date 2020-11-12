#include <device/tensor_view.hpp>
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"

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

    EXPECT_EQ(24, d->dimensions());
  }

}