#include <gtest/gtest.h>
#include "layers/linear.hpp"
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include "ops/conversions.hpp"
#include "optimizers/adam.hpp"
#include <losses/crossentropyloss.hpp>
#include "layers/input.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class AdamColMajor : public ::testing::Test {
  protected:
    void SetUp() override {
      cd.init();
      lr = 0.001;
      new_bias.resize(cd.out_dims[1]);
      new_weights.resize(cd.weights_dim_shuffled);
    }

    auto PropagateGradient(int epochs) {

      // input layer
      Input<float, ThreadPoolDevice> input;
      input.set_input(cd.linearInput);

      // create fully connected layer
      Linear<float, ThreadPoolDevice, ColMajor> linear(cd.dims[1], cd.out_dims[1]);
      linear.init(cd.weights.to_host());

      // create loss function
      CrossEntropyLoss<float, float, 2, ThreadPoolDevice, ColMajor> loss_func;

      // create  optimizer
      EigenSinn::Adam<float, 2> adam(lr);
      
      for (int i = 0; i < epochs; i++) {

        // propagate forward through the model
        linear.forward(input);

        DeviceTensor<float, 2, ThreadPoolDevice, ColMajor> output(linear.get_output());

        // compute loss
        loss_func.step(output, cd.target);
        DeviceTensor<float, 2, ThreadPoolDevice, ColMajor> dloss(loss_func.get_loss_derivative_by_input());

        // propagate back through the fc layer
        // compute dL/dw, dL/db, dL/dx
        linear.backward(input, dloss.raw());

        adam.step(linear);
      }

      EXPECT_TRUE(is_elementwise_approx_eq(new_weights, linear.get_weights()));
      EXPECT_TRUE(is_elementwise_approx_eq(new_bias, linear.get_bias()));
    }

    DeviceTensor<float, 2, ThreadPoolDevice, ColMajor> new_weights;
    DeviceTensor<float, 1, ThreadPoolDevice, ColMajor> new_bias;
    CommonData2d<ThreadPoolDevice, ColMajor> cd;
    float lr;
  };


  TEST_F(AdamColMajor, OneStep) {

    DeviceTensor<float, 2, ThreadPoolDevice, ColMajor> tmp(cd.weight_dims);
    tmp.setValues({ { 0.30941489,  0.16201581,  0.06012393,  0.32472005, -0.00491815,
         -0.07433553,  0.16475038, -0.35374174},
        { 0.19189887, -0.24621475,  0.27166173, -0.00426837, -0.18301390,
         -0.20750742, -0.27948126,  0.29742351},
        {-0.15596132,  0.14989461,  0.16839565, -0.25125238, -0.18178347,
         -0.07753529, -0.32977933,  0.19727282},
        {-0.28225577, -0.15681732, -0.32588497, -0.08420141, -0.27585772,
         -0.02888693,  0.18839149,  0.32116404} });

    new_weights.view() = tmp->shuffle(array<Index, 2>{1, 0});
    new_bias.setValues({ -0.00100000,  0.00100000,  0.00100000, -0.00100000 });

    PropagateGradient(1);
  }

  TEST_F(AdamColMajor, TwoSteps) {

    DeviceTensor<float, 2, ThreadPoolDevice, ColMajor> tmp(cd.weight_dims);
    tmp.setValues({ { 0.31041464,  0.16101657,  0.06112371,  0.32372031, -0.00391832,
         -0.07533528,  0.16575015, -0.35474178},
        { 0.19289874, -0.24721453,  0.27266166, -0.00326850, -0.18201400,
         -0.20850728, -0.27848139,  0.29842332},
        {-0.15696105,  0.14889462,  0.16739571, -0.25225237, -0.18278342,
         -0.07653541, -0.33077928,  0.19827282},
        {-0.28325558, -0.15581743, -0.32688382, -0.08320152, -0.27485764,
         -0.02788713,  0.18939233,  0.32016420} });

    new_weights.view() = tmp->shuffle(array<Index, 2>{1, 0});
    new_bias.setValues({ -0.00199981,  0.00199963,  0.00199995, -0.00199998 });
    
    PropagateGradient(2);
  }
}