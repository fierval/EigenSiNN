#include <gtest/gtest.h>
#include "layers/linear.hpp"
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "layers/input.hpp"
#include "gpu/gpu_tensor.hpp"

using namespace Eigen;

namespace EigenSinnTest {
  class LinearGpu : public ::testing::Test {

  protected:
    void SetUp() override {
      cd.init();

    }

    CommonData2d cd;
  };

  TEST_F(LinearGpu, Backward) {
    EigenSinn::Dispatcher<GpuDevice> device;
    float* d_input = EigenSinn::to_device(cd.linearInput);

    EigenSinn::Input<float, 2, GpuDevice> input(cd.dims, device);
    input.set_input(d_input);

  }
}