#include <gtest/gtest.h>
#include "device/device_tensor.hpp"
#include <helpers/cudnn_workspace.hpp>

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

      params = std::make_shared<ConvolutionParams<4>>(cd.convInput.dimensions(), cd.convWeights.dimensions(), padding, stride, dilation, false);
      W = std::make_shared<CudnnWorkspace>(device(), *params);
    }

    CommonData4d<CudnnDevice, RowMajor> cd;
    ConvDataWith1Padding<CudnnDevice, RowMajor> cd1p;

    const int stride = 1, dilation = 1;
    const Padding2D padding{ 0, 0 };

    std::shared_ptr<ConvolutionParams<4> >params;
    static inline DeviceWrapper<CudnnDevice> device;

    std::shared_ptr<CudnnWorkspace> W;
  };

  TEST_F(CudnnTest, SimpleConvForward) {

    //Create all the descriptors
    // - cudnn
    // - input tensor
    // - filter
    // - convolution
    DeviceTensor<float, 4, CudnnDevice, RowMajor> out(params->output_dims());

    // forward convolution
    checkCudnnErrors(cudnnConvolutionForward(W->cudnn(), &(W->one), W->input_desc, cd.convInput->data(),
      W->filter_desc, cd.convWeights->data(), W->conv_desc, W->conv_fwd_algo, W->d_workspace, W->workspace_size,
      &(W->zero), W->output_desc, out->data()));

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, out));
   
  }

  TEST_F(CudnnTest, SimpleConvBackward) {

    //Create all the descriptors
    // - cudnn
    // - input tensor
    // - filter
    // - convolution

    DeviceTensor<float, 4, CudnnDevice, RowMajor> out(params->output_dims());
    DeviceTensor<float, 4, CudnnDevice, RowMajor> dinput(params->orig_dims());
    DeviceTensor<float, 4, CudnnDevice, RowMajor> dweight(params->kernel_dims);

    // forward convolution
    checkCudnnErrors(cudnnConvolutionForward(W->cudnn(), &(W->one), W->input_desc, cd.convInput->data(),
      W->filter_desc, cd.convWeights->data(), W->conv_desc, W->conv_fwd_algo, W->d_workspace, W->workspace_size,
      &(W->zero), W->output_desc, out->data()));

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, out));

    // data backwards
    checkCudnnErrors(cudnnConvolutionBackwardData(W->cudnn(), &(W->one), W->filter_desc, cd.convWeights->data(), 
      W->output_desc, cd.convLoss->data(), W->conv_desc, W->conv_bwd_data_algo, W->d_workspace, W->workspace_size, &(W->zero), W->input_desc, dinput->data()));


    EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, dinput));

    // weights backwards
    checkCudnnErrors(
      cudnnConvolutionBackwardFilter(W->cudnn(), &(W->one), W->input_desc, cd.convInput->data(), W->output_desc, cd.convLoss->data(),
        W->conv_desc, W->conv_bwd_filter_algo, W->d_workspace, W->workspace_size, &(W->zero), W->filter_desc, dweight->data()));

    EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, dweight));
  }

}