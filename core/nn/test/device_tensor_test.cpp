#include <device/tensor_view.hpp>
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"

#include "gtest/gtest.h"

using namespace EigenSinn;

namespace EigenSinnTest {

  class DeviceTensor : public ::testing::Test {

  protected:
    CommonData4d cd;
  };

  TEST_F(DeviceTensor, CreateVariadicNoValue) {

  }
}