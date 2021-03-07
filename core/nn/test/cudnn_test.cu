#include <gtest/gtest.h>
#include "device/device_tensor.hpp"
#include <cudnn/helper.hpp>

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

    const int padding = 0, stride = 1, dilation = 1;
    DSizes<Index, 4> output_size;

  };

#if 0
  TEST_F(CudnnTest, SimpleConv) {

    //Create all the descriptors
    // - cudnn
    // - input tensor
    // - filter
    // - convolution
    CudaContext ctx;

    ctx.input_desc = tensor4d(cd.convInput.dimensions());
    
    // Filter properties
    cudnnCreateFilterDescriptor(&ctx.filter_desc);

    // Convolution descriptor, set properties
    cudnnCreateConvolutionDescriptor(&ctx.conv_desc);
    checkCudnnErrors(cudnnSetConvolution2dDescriptor(ctx.conv_desc, padding, padding, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // setting cudnn convolution math type
    // CUDNN_DEFAULT_MATH operates convolution with FP32.
    // If you use A100, CUDNN utilise tensor cores with TF32.
    checkCudnnErrors(cudnnSetConvolutionMathType(ctx.conv_desc, CUDNN_DEFAULT_MATH));

    // get convolution output dimensions
    output_size = set_output_dims(ctx.conv_desc, ctx.input_desc, ctx.filter_desc);
    DeviceTensor<float, 4, GpuDevice, RowMajor> out(output_size);

    // create output tensor descriptor
    ctx.output_desc = tensor4d(output_size);

    // forward convolution
    checkCudnnErrors(cudnnConvolutionForward(ctx.cudnn(), &ctx.one, ctx.input_desc, cd.convInput->data(),
      ctx.filter_desc, cd.convWeights->data(), ctx.conv_desc, ctx.conv_fwd_algo, ctx.d_workspace, ctx.workspace_size,
      &ctx.zero, ctx.output_desc, out->data()));

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, out));
    ctx.set_workspace();
   
  }
#endif
}