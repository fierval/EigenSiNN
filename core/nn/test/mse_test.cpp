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
      loss = 0.50550109;
      cd.init();
    }

    float loss;
    CommonData2d cd;
  };

  TEST_F(Mse, Compute) {

    Linear<float> fc(cd.dims[0], cd.dims[1], cd.out_dims[1], false);
    
    fc.init(cd.weights);
    fc.forward(cd.linearInput);

    Tensor<float, 2> output = from_any<float, 2>(fc.get_output());

    EXPECT_TRUE(is_elementwise_approx_eq(fc.get_output(), cd.output));

    MseLoss<float, 2> mse;
    mse.compute(cd.target, output);

    EXPECT_EQ(loss, from_any_scalar<float>(mse.get_output()));
  }
}
