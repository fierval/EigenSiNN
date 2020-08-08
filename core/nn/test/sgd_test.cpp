#include <gtest/gtest.h>
#include "layers/linear.hpp"
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include "optimizers/sgd.hpp"
#include <losses\crossentropyloss.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class SGD : public ::testing::Test {
  protected:
    void SetUp() override {
      cd.init();
      lr = 0.1;

      Tensor<float, 2> tmp(cd.weight_dims);
      tmp.setValues({ { 0.32386121,  0.16082032,  0.08126653,  0.30421251, -0.00304944,
           -0.08350309,  0.16417494, -0.35673440},
          { 0.20838137, -0.26138818,  0.29403499, -0.00155395, -0.17344446,
           -0.22037405, -0.27832744,  0.32912356},
          {-0.15800315,  0.12336881,  0.12830539, -0.27609718, -0.20152104,
           -0.07332371, -0.33157253,  0.20261464},
          {-0.31114277, -0.11192260, -0.32931057, -0.04156353, -0.26955828,
           -0.01106431,  0.18760630,  0.28711483} });

      new_weights = tmp.shuffle(array<Index, 2>{1, 0});
    }

    Tensor<float, 2> new_weights;
    CommonData2d cd;
    float lr;
  };

  TEST_F(SGD, ZeroMomentum) {

    // create and initialize an fc layer
    Linear<float> linear(cd.dims[0], cd.dims[1], cd.out_dims[1]);

    linear.init(cd.weights);
    linear.forward(cd.linearInput);
    
    auto output = linear.get_output();

    // create loss function
    // propagate loss and get its gradient
    CrossEntropyLoss<float> loss_func;

    loss_func.forward(output, cd.target);
    loss_func.backward();

    auto doutput = loss_func.get_loss_derivative_by_input();

    // propagate back through the fc layer
    linear.backward(cd.linearInput, doutput);

    // collect weights and biases and perform the GD step
    auto weights = linear.get_weights(), dweights = linear.get_loss_by_weights_derivative();
    auto bias = linear.get_bias(), dbias = linear.get_loss_by_bias_derivative();

    EigenSinn::SGD<float, 2> sgd(lr);

    std::tie(weights, bias) = sgd.step(weights, bias, dweights, dbias);

    EXPECT_TRUE(is_elementwise_approx_eq(new_weights, weights));
  }
}