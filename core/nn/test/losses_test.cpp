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

      inp = new Input<float, 2>(cd.dims);
      inp->set_input(cd.linearInput.data());

      fc = new Linear<float>(cd.dims[1], cd.out_dims[1]);

      fc->init(cd.weights);
      fc->forward(*inp);
    }

    void TearDown() {
      delete fc;
      delete inp;
    }

    float loss;
    Tensor<float, 2> dloss;
    CommonData2d cd;
    Linear<float>* fc;
    Input<float, 2> * inp;
  };

  TEST_F(Loss, MSE) {
    
    loss = 0.50550109;

    dloss.setValues({ {-0.10515201,  0.08070742,  0.01504049,  0.08132496},
      { 0.04065431, -0.21157818, -0.09260463, -0.06111295},
      {-0.12644342,  0.00961572, -0.17200474,  0.19923662} });

    TensorMap<Tensor<float, 2>> output(fc->get_output(), vector2array<2>(fc->get_out_dims()));

    MseLoss<float, float, 2> loss_func;
    loss_func.step(output, cd.target);

    EXPECT_EQ(loss, loss_func.get_output());
    EXPECT_TRUE(is_elementwise_approx_eq(dloss, loss_func.get_loss_derivative_by_input()));
  }

  TEST_F(Loss, CrossEntropy) {

    loss = 1.03314781;

    dloss.setValues({ { 0.02313359, -0.14153539,  0.04758135,  0.07082045},
              { 0.10596204,  0.02332874,  0.04763307, -0.17692387},
              { 0.02690827,  0.06087292, -0.27768427,  0.18990307} });

    TensorMap<Tensor<float, 2>> output(fc->get_output(), vector2array<2>(fc->get_out_dims()));

    CrossEntropyLoss<float, float, 2> loss_func;
    loss_func.step(output, cd.target);

    EXPECT_EQ(loss, loss_func.get_output());
    EXPECT_TRUE(is_elementwise_approx_eq(dloss, loss_func.get_loss_derivative_by_input()));
  }

}
