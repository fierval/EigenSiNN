#include <gtest/gtest.h>
#include <layers/batchnorm.hpp>
#include <layers/input.hpp>
#include <iostream>
#include <ops/conversions.hpp>
#include <ops/comparisons.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class Batchnorm1dTest : public ::testing::Test {
  protected:
    void SetUp() override {

      MatrixXf input_matrix, loss_matrix;
      input_matrix.resize(batch_size, cols);
      loss_matrix.resize(batch_size, cols);

      loss_matrix << 0.07951424, 0.39795890, 0.48816258, 0.58650136, 0.80818069,
        0.33679566, 0.74452204, 0.24355969, 0.36228219, 0.69534987;

      input_matrix << 0.56279999, 0.93430001, 0.25929999, 0.79210001, 0.15889999,
        0.18510000, 0.04310000, 0.50970000, 0.88209999, 0.58310002;

      input = Matrix_to_Tensor(input_matrix, batch_size, cols);
      loss = Matrix_to_Tensor(loss_matrix, batch_size, cols);
      gamma.resize(cols);
      beta.resize(cols);

      output.resize(input.dimensions());
      output.setValues({ { 1.09985983, 2.19994974, -2.69904375, -3.59016252, -4.49944401},
        {-0.89985991, -1.79994965, 3.29904366, 4.39015722, 5.49944496} });

      gamma.setValues({1., 2.0, 3., 4., 5.});
      beta = gamma * 0.1f;
    }

    //void TearDown() override {}

    Tensor<float, 2> input, loss, output;
    TensorSingleDim<float> beta, gamma;
    const float eps = 1e-5, momentum = 0.9;
    const Index batch_size = 2, cols = 5;
  };

  TEST_F(Batchnorm1dTest, Backward) {

    Input<float, 2> input_layer;
    input_layer.set_input(input);

    BatchNormalizationLayer<float, 2> bn(cols, eps, momentum);

    DeviceTensor<float, 2> expected_derivative(batch_size, cols);
    DeviceTensor<float, 1> exp_dbeta(cols), exp_dgamma(cols);

    exp_dbeta.setValues({ 0.41630989, 1.14248097, 0.73172224, 0.94878352, 1.50353050 });
    exp_dgamma.setValues({ -0.25724539, -0.34655440, -0.24452491, -0.22366823, -0.11281815 });

    expected_derivative.setValues({{-1.90824110e-04, -3.91245958e-05, 1.86752446e-03, 4.88124043e-02, 2.96708080e-04}, 
      {1.90745224e-04, 3.91245958e-05, -1.86752446e-03, -4.88256179e-02, -2.96444632e-04 }});

    bn.init(beta, gamma);
    bn.forward(input_layer);

    EXPECT_TRUE(is_elementwise_approx_eq(DeviceTensor<float, 2>(output), bn.get_output()));

    DeviceTensor<float, 2> loss_device(loss);
    bn.backward(input_layer, loss_device.raw());

    EXPECT_TRUE((is_elementwise_approx_eq<float, 2>(expected_derivative, bn.get_loss_by_input_derivative(), 4e-5)));
    EXPECT_TRUE((is_elementwise_approx_eq<float, 1>(exp_dbeta, bn.get_loss_by_bias_derivative(), 1e-5)));
    EXPECT_TRUE((is_elementwise_approx_eq<float, 1>(exp_dgamma, bn.get_loss_by_weights_derivative(), 1e-5)));
  } 
}