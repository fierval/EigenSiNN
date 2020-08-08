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

    auto PropagateGradient(int epochs, float momentum = 0.0, bool nesterov = false) {
      // create fully connected layer
      Linear<float> linear(cd.dims[0], cd.dims[1], cd.out_dims[1]);
      linear.init(cd.weights);

      // create loss function
      CrossEntropyLoss<float> loss_func;
      Tensor<float, 2> weights, dweights;
      Tensor<float, 1> bias, dbias;

      // create  optimizer
      EigenSinn::SGD<float, 2> sgd(lr, momentum, nesterov);
      
      for (int i = 0; i < epochs; i++) {

        // propagate forward through the model
        linear.forward(cd.linearInput);

        auto output = linear.get_output();

        // compute loss
        loss_func.forward(output, cd.target);

        // start propagating back
        // 1. compute dL/dy
        loss_func.backward();

        auto dloss = loss_func.get_loss_derivative_by_input();

        // propagate back through the fc layer
        // compute dL/dw, dL/db, dL/dx
        linear.backward(cd.linearInput, dloss);

        // collect weights and biases and perform the GD step
        auto weights_auto = linear.get_weights(), dweights_auto = linear.get_loss_by_weights_derivative();
        auto bias_auto = linear.get_bias(), dbias_auto = linear.get_loss_by_bias_derivative();

        //std::any new_weights, new_bias;
        std::tie(weights_auto, bias_auto) = sgd.step(weights_auto, bias_auto, dweights_auto, dbias_auto);

        // set new weights and biases in the layer
        linear.set_weights(weights_auto);
        linear.set_bias(bias_auto);
      }

      return std::make_tuple(linear.get_weights(), linear.get_bias());
    }

    Tensor<float, 2> new_weights;
    CommonData2d cd;
    float lr;
  };

  TEST_F(SGD, ZeroMomentum) {

    std::any weights, bias;
    std::tie(weights, bias) = PropagateGradient(1, 0, false);

    EXPECT_TRUE(is_elementwise_approx_eq(new_weights, weights));
  }
}