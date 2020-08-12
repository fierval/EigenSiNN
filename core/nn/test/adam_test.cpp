#include <gtest/gtest.h>
#include "layers/linear.hpp"
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"
#include "optimizers/adam.hpp"
#include <losses\crossentropyloss.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class Adam : public ::testing::Test {
  protected:
    void SetUp() override {
      cd.init();
      lr = 0.01;
      new_bias.resize(cd.out_dims[1]);
    }

    auto PropagateGradient(int epochs) {
      // create fully connected layer
      Linear<float> linear(cd.dims[0], cd.dims[1], cd.out_dims[1]);
      linear.init(cd.weights);

      // create loss function
      CrossEntropyLoss<float> loss_func;

      // create  optimizer
      EigenSinn::Adam<float, 2> adam(lr);
      
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
        std::tie(weights_auto, bias_auto) = adam.step(weights_auto, bias_auto, dweights_auto, dbias_auto);

        // set new weights and biases in the layer
        linear.set_weights(weights_auto);
        linear.set_bias(bias_auto);
      }

      return std::make_tuple(linear.get_weights(), linear.get_bias());
    }

    void RunPropTest(int epochs) {
      std::any weights, bias;
      std::tie(weights, bias) = PropagateGradient(epochs);

      EXPECT_TRUE(is_elementwise_approx_eq(new_weights, weights));
      EXPECT_TRUE(is_elementwise_approx_eq(new_bias, bias));

    }

    Tensor<float, 2> new_weights;
    Tensor<float, 1> new_bias;
    CommonData2d cd;
    float lr;
  };


  TEST_F(Adam, OneStep) {

    Tensor<float, 2> tmp(cd.weight_dims);
    tmp.setValues({ { 0.30941489,  0.16201581,  0.06012393,  0.32472005, -0.00491815,
         -0.07433553,  0.16475038, -0.35374174},
        { 0.19189887, -0.24621475,  0.27166173, -0.00426837, -0.18301390,
         -0.20750742, -0.27948126,  0.29742351},
        {-0.15596132,  0.14989461,  0.16839565, -0.25125238, -0.18178347,
         -0.07753529, -0.32977933,  0.19727282},
        {-0.28225577, -0.15681732, -0.32588497, -0.08420141, -0.27585772,
         -0.02888693,  0.18839149,  0.32116404} });

    new_weights = tmp.shuffle(array<Index, 2>{1, 0});
    new_bias.setValues({ -0.00100000,  0.00100000,  0.00100000, -0.00100000 });

    RunPropTest(1);
  }

  //TEST_F(Adam, Momentum1Step) {

  //  Tensor<float, 2> tmp(cd.weight_dims);
  //  tmp.setValues({ { 3.37926388e-01,  1.59690693e-01,  1.02176636e-01,  2.84526557e-01,
  //       -2.26725824e-04, -9.27660763e-02,  1.64573103e-01, -3.61379057e-01},
  //      { 2.25439876e-01, -2.75997013e-01,  3.18087339e-01,  2.55925790e-03,
  //       -1.62596002e-01, -2.33967990e-01, -2.76166409e-01,  3.60174716e-01},
  //      {-1.61181465e-01,  9.41504315e-02,  8.47780257e-02, -3.03573787e-01,
  //       -2.23511130e-01, -6.78276271e-02, -3.34533960e-01,  2.09348083e-01},
  //      {-3.39088142e-01, -6.69657737e-02, -3.30745667e-01,  1.48580968e-03,
  //       -2.61239380e-01,  6.29654154e-03,  1.88008562e-01,  2.53974885e-01} });

  //  new_weights = tmp.shuffle(array<Index, 2>{1, 0});
  //  new_bias.setValues({ -0.03066470,  0.01040321,  0.03756850, -0.01730700 });
  //  
  //  RunPropTest(2);
  //}

  //TEST_F(Adam, Momentum1StepNesterov) {

  //  Tensor<float, 2> tmp(cd.weight_dims);
  //  tmp.setValues({ { 3.39028239e-01,  1.59707010e-01,  1.03905559e-01,  2.82974273e-01,
  //        2.06110999e-05, -9.34924558e-02,  1.64605826e-01, -3.61862838e-01},
  //      { 2.26916432e-01, -2.77127922e-01,  3.20312917e-01,  2.97543406e-03,
  //       -1.61594197e-01, -2.35152304e-01, -2.75972277e-01,  3.62768531e-01},
  //      {-1.61484808e-01,  9.13537145e-02,  8.06158632e-02, -3.06197733e-01,
  //       -2.25612834e-01, -6.73028156e-02, -3.34816873e-01,  2.09998086e-01},
  //      {-3.41363192e-01, -6.30544350e-02, -3.30538005e-01,  5.24589792e-03,
  //       -2.60386795e-01,  7.68242404e-03,  1.88064635e-01,  2.51214862e-01} });

  //  new_weights = tmp.shuffle(array<Index, 2>{1, 0});

  //  new_bias.setValues({ -0.03194755,  0.01070375,  0.03941960, -0.01817579 });
  //  
  //  RunPropTest(2);
  //}

}