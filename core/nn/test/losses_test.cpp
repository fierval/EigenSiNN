#include <iostream>
#include <gtest/gtest.h>
#include <layers/linear.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"
#include "losses/mse.hpp"
#include "include/testutils.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Mse : public ::testing::Test {
  protected:
    void SetUp() {
      
      cd.init();
      doutput.resize(cd.out_dims);

      fc = new Linear<float>(cd.dims[0], cd.dims[1], cd.out_dims[1], false);

      fc->init(cd.weights);
      fc->forward(cd.linearInput);

      EXPECT_TRUE(is_elementwise_approx_eq(fc->get_output(), cd.output));

    }

    void TearDown() {
      delete fc;
    }

    float loss;
    Tensor<float, 2> doutput;
    CommonData2d cd;
    Linear<float>* fc;
  };

  TEST_F(Mse, Compute) {
    
    loss = 0.50550109;

    doutput.setValues({ {-0.10515201,  0.08070742,  0.01504049,  0.08132496},
      { 0.04065431, -0.21157818, -0.09260463, -0.06111295},
      {-0.12644342,  0.00961572, -0.17200474,  0.19923662} });

    Tensor<float, 2> output = from_any<float, 2>(fc->get_output());

    MseLoss<float> mse;
    mse.forward(output, cd.target);

    EXPECT_EQ(loss, from_any_scalar<float>(mse.get_output()));
  }
}
