#include <gtest/gtest.h>
#include "device/device_tensor.hpp"
#include <helpers/cudnn_helpers.hpp>

#include "include/commondata4d.hpp"
#include "include/convdata4d.hpp"
#include "ops/comparisons.hpp"

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

    const int stride = 1, dilation = 1;
    const Padding2D padding{ 0, 0 };

  };

  TEST_F(CudnnTest, SimpleConvForward) {

    //Create all the descriptors
    // - cudnn
    // - input tensor
    // - filter
    // - convolution
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;

    ConvolutionParams<4> params(cd.convInput.dimensions(), cd.convWeights.dimensions(), padding, stride, dilation, false);
    CudnnWorkspace ctx(params);

    DeviceTensor<float, 4, GpuDevice, RowMajor> out(params.output_dims());

    // forward convolution
    checkCudnnErrors(cudnnConvolutionForward(ctx.cudnn(), &ctx.one, ctx.input_desc, cd.convInput->data(),
      ctx.filter_desc, cd.convWeights->data(), ctx.conv_desc, ctx.conv_fwd_algo, ctx.d_workspace, ctx.workspace_size,
      &ctx.zero, ctx.output_desc, out->data()));

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, out));
   
  }
}