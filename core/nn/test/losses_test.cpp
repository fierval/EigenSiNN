#include <iostream>
#include <gtest/gtest.h>
#include <layers/linear.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"
#include "losses/mse.hpp"
#include "losses/crossentropyloss.hpp"
#include "include/testutils.hpp"
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class Loss : public ::testing::Test {
  protected:
    void SetUp() {
      
      cd.init();
      dloss.resize(cd.out_dims);

      inp.set_input(cd.linearInput);

      fc = new Linear<float>(cd.dims[1], cd.out_dims[1]);

      fc->init(cd.weights.to_host());
      fc->forward(inp.get_output());
    }

    void TearDown() {
      delete fc;
    }

    float loss;
    DeviceTensor<float, 2> dloss;
    CommonData2d<ThreadPoolDevice> cd;
    Linear<float>* fc;
    Input<float> inp;
  };

  TEST_F(Loss, MSE) {
    
    loss = 0.50550109;

    dloss.setValues({ {-0.10515201,  0.08070742,  0.01504049,  0.08132496},
      { 0.04065431, -0.21157818, -0.09260463, -0.06111295},
      {-0.12644342,  0.00961572, -0.17200474,  0.19923662} });

    DeviceTensor<float, 2> output(fc->get_output());

    MseLoss<float, float, 2> loss_func;
    loss_func.step(output.raw(), cd.target.raw());

    EXPECT_TRUE(std::abs(loss_func.get_output() - loss) < 1e-5);
    EXPECT_TRUE(is_elementwise_approx_eq(dloss, loss_func.get_loss_derivative_by_input()));
  }

  TEST_F(Loss, CrossEntropy) {

    loss = 1.03314781;

    dloss.setValues({ { 0.02313359, -0.14153539,  0.04758135,  0.07082045},
              { 0.10596204,  0.02332874,  0.04763307, -0.17692387},
              { 0.02690827,  0.06087292, -0.27768427,  0.18990307} });

    DeviceTensor<float, 2> output(fc->get_output());

    CrossEntropyLoss<float, float, 2> loss_func;
    loss_func.step(output.raw(), cd.target.raw());

    EXPECT_EQ(loss, loss_func.get_output());
    EXPECT_TRUE(is_elementwise_approx_eq(dloss, loss_func.get_loss_derivative_by_input()));
  }

}
