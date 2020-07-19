#include <gtest/gtest.h>
#include <iostream>
#include "layers/convolution.hpp"
#include "include/commondata4d.hpp"
#include "ops/comparisons.hpp"
#include "ops/im2col.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Convolution : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
    }

    CommonData4d cd;
  };

  TEST_F(Convolution, Forward) {
    
    Conv2d<float> conv2d(cd.kernelDims);

    conv2d.init(cd.convWeights);
    conv2d.forward(cd.convInput);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.output, conv2d.get_output()));
  }

  //TEST_F(Convolution, Backward) {

  //  Conv2d<float> conv2d(cd.kernelDims);

  //  conv2d.init(cd.convWeights);
  //  conv2d.forward(cd.convInput);
  //  conv2d.backward(cd.convInput, cd.convLoss);

  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dinput, conv2d.get_loss_by_input_derivative()));
  //  EXPECT_TRUE(is_elementwise_approx_eq(cd.dweight, conv2d.get_loss_by_weights_derivative()));
  //}


  TEST_F(Convolution, im2col) {

    Tensor<float, 3> chip = cd.convInput.chip(0, 0);

    std::cerr << chip << std::endl <<std::endl << "################################################" << std::endl << std::endl;

    float* out_data = new float[cd.dims[1] * cd.kernelDims[3] * cd.kernelDims[2] 
      * cd.convOutDims[2] * cd.convOutDims[3]];

    im2col(chip.data(), cd.dims[1], cd.dims[2], cd.dims[3], cd.convOutDims[2], cd.convOutDims[3], 
      cd.kernelDims[2], cd.kernelDims[3], 0, 0, 1, 1, 1, 1, out_data);

    TensorMap<Tensor<float, 2>> out_tensor(out_data, cd.dims[1] * cd.kernelDims[3] * cd.kernelDims[2], cd.convOutDims[2] * cd.convOutDims[3]);

    std::cerr << out_tensor << std::endl << std::endl;
    delete [] out_data;
  }
}