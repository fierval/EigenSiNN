#include <gtest/gtest.h>
#include "layers/linear.hpp"
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include "optimizers/sgd.hpp"
#include <losses\crossentropyloss.hpp>
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class SGDRowMajor : public ::testing::Test {
  protected:
    void SetUp() override {
      cd.init();
      lr = 0.1;
      new_bias.resize(cd.out_dims[1]);
      new_weights.resize(cd.weights_dim_shuffled);
    }

    auto PropagateGradient(int epochs, float momentum = 0.0, bool nesterov = false) {

      Input<float, 2, ThreadPoolDevice, RowMajor> input;
      input.set_input(cd.linearInput);

      // create fully connected layer
      Linear<float, ThreadPoolDevice, RowMajor> linear(cd.dims[1], cd.out_dims[1]);
      linear.init(cd.weights.to_host());

      // create loss function
      CrossEntropyLoss<float, float, 2, ThreadPoolDevice, RowMajor> loss_func;

      // create  optimizer
      EigenSinn::SGD<float, 2> sgd(lr, momentum, nesterov);
      
      for (int i = 0; i < epochs; i++) {

        // propagate forward through the model
        linear.forward(input);

        DeviceTensor<float, 2, ThreadPoolDevice, RowMajor> output(linear.get_output());

        // compute loss
        // start propagating back
        // 1. compute dL/dy
        loss_func.step(output, cd.target);

        DeviceTensor<float, 2, ThreadPoolDevice, RowMajor> dloss(loss_func.get_loss_derivative_by_input());

        // propagate back through the fc layer
        // compute dL/dw, dL/db, dL/dx
        linear.backward(input, dloss.raw());

        PtrTensorAdapter<float, ThreadPoolDevice> weights_any, bias_any;
        sgd.step(linear);
      }

      EXPECT_TRUE(is_elementwise_approx_eq(new_weights, linear.get_weights()));
      EXPECT_TRUE(is_elementwise_approx_eq(new_bias, linear.get_bias()));

    }

    DeviceTensor<float, 2, ThreadPoolDevice, RowMajor> new_weights;
    DeviceTensor<float, 1, ThreadPoolDevice, RowMajor> new_bias;
    CommonData2d<ThreadPoolDevice, RowMajor> cd;
    float lr;
  };


  TEST_F(SGDRowMajor, ZeroMomentum) {

    DeviceTensor<float, 2, ThreadPoolDevice, RowMajor> tmp(cd.weight_dims);
    tmp.setValues({ { 0.32386121,  0.16082032,  0.08126653,  0.30421251, -0.00304944,
         -0.08350309,  0.16417494, -0.35673440},
        { 0.20838137, -0.26138818,  0.29403499, -0.00155395, -0.17344446,
         -0.22037405, -0.27832744,  0.32912356},
        {-0.15800315,  0.12336881,  0.12830539, -0.27609718, -0.20152104,
         -0.07332371, -0.33157253,  0.20261464},
        {-0.31114277, -0.11192260, -0.32931057, -0.04156353, -0.26955828,
         -0.01106431,  0.18760630,  0.28711483} });

    new_weights.view() = tmp->shuffle(array<Index, 2>{1, 0});
    new_bias.setValues({ -0.01560039,  0.00573337,  0.01824698, -0.00837997 });

    PropagateGradient(1, 0, false);
  }

  TEST_F(SGDRowMajor, Momentum1Step) {

    DeviceTensor<float, 2, ThreadPoolDevice, RowMajor> tmp(cd.weight_dims);
    tmp.setValues({ { 3.37926388e-01,  1.59690693e-01,  1.02176636e-01,  2.84526557e-01,
         -2.26725824e-04, -9.27660763e-02,  1.64573103e-01, -3.61379057e-01},
        { 2.25439876e-01, -2.75997013e-01,  3.18087339e-01,  2.55925790e-03,
         -1.62596002e-01, -2.33967990e-01, -2.76166409e-01,  3.60174716e-01},
        {-1.61181465e-01,  9.41504315e-02,  8.47780257e-02, -3.03573787e-01,
         -2.23511130e-01, -6.78276271e-02, -3.34533960e-01,  2.09348083e-01},
        {-3.39088142e-01, -6.69657737e-02, -3.30745667e-01,  1.48580968e-03,
         -2.61239380e-01,  6.29654154e-03,  1.88008562e-01,  2.53974885e-01} });

    new_weights.view() = tmp->shuffle(array<Index, 2>{1, 0});
    new_bias.setValues({ -0.03066470,  0.01040321,  0.03756850, -0.01730700 });
    
    PropagateGradient(2, 0.1, false);
  }

  TEST_F(SGDRowMajor, Momentum1StepNesterov) {

    DeviceTensor<float, 2, ThreadPoolDevice, RowMajor> tmp(cd.weight_dims);
    tmp.setValues({ { 3.39028239e-01,  1.59707010e-01,  1.03905559e-01,  2.82974273e-01,
          2.06110999e-05, -9.34924558e-02,  1.64605826e-01, -3.61862838e-01},
        { 2.26916432e-01, -2.77127922e-01,  3.20312917e-01,  2.97543406e-03,
         -1.61594197e-01, -2.35152304e-01, -2.75972277e-01,  3.62768531e-01},
        {-1.61484808e-01,  9.13537145e-02,  8.06158632e-02, -3.06197733e-01,
         -2.25612834e-01, -6.73028156e-02, -3.34816873e-01,  2.09998086e-01},
        {-3.41363192e-01, -6.30544350e-02, -3.30538005e-01,  5.24589792e-03,
         -2.60386795e-01,  7.68242404e-03,  1.88064635e-01,  2.51214862e-01} });

    new_weights.view() = tmp->shuffle(array<Index, 2>{1, 0});

    new_bias.setValues({ -0.03194755,  0.01070375,  0.03941960, -0.01817579 });
    
    PropagateGradient(2, 0.1, true);
  }

}