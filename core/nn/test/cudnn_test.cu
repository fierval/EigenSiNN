#include <gtest/gtest.h>
#include "device/device_tensor.hpp"
#include <cudnn/helper.hpp>

#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"

using namespace Eigen;
using namespace EigenSinn;

namespace EigenSinnTest {
  class CudnnTest : public ::testing::Test {

  protected:
    void SetUp() override {
      cd.init();
      cd1p.init();
    }

    CommonData4d<GpuDevice, RowMajor> cd;
    ConvDataWith1Padding<GpuDevice, RowMajor> cd1p;

    const Padding2D padding = { 0, 0 };


  };

  TEST_F(CudnnTest, Simple) {

    cudnnHandle_t cudnnHandle;

    checkCudnnErrors(cudnnCreate(&cudnnHandle));
    cudnnDestroy(cudnnHandle);
  }
}