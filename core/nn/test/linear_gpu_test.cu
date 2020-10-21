#include <gtest/gtest.h>
#include "layers/linear.hpp"
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "layers/input.hpp"
#include "gpu/gpu_tensor.hpp"

using namespace Eigen;
using namespace EigenSinn;

namespace EigenSinnTest {
  class LinearGpu : public ::testing::Test {

  protected:
    void SetUp() override {
      cd.init();

    }

    CommonData2d cd;
  };

  TEST_F(LinearGpu, Backward) {
    Dispatcher<GpuDevice> device;
    float* d_input = to_device(cd.linearInput);

    Input<float, 2, GpuDevice> input(cd.dims, device);
    input.set_input(d_input);

    Linear<float, GpuDevice> linear(cd.dims[1], cd.out_dims[1]);

  }
}