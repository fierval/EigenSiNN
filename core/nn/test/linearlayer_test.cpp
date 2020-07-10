#include "layers/input.hpp"
#include "layers/linear.hpp"
#include "Random.h"
#include <iostream>
#include "ops/conversions.hpp"

#include <gtest/gtest.h>


using namespace EigenSinn;

namespace EigenSinnTest {

  TEST(FullyConnected, BackpropTest) {
    const int linear_layer_out = 3, batch_size = 2, linear_layer_in = 4;
    MatrixXf input_matrix;
    input_matrix.resize(batch_size, linear_layer_in);
    input_matrix <<
      1, 2, 3, 4,
      5, 6, 7, 8;

    //initialization constants
    RNG rng(1);
    double mu = 0;
    double sigma = 0.01;

    LinearTensor input_tensor = Matrix_to_Tensor(input_matrix, batch_size, linear_layer_in);
    InputLayer<float, 2> input(input_tensor);
    Linear linear(input.batch_size(), input.input_vector_dim(), linear_layer_out, false, true);

    linear.init();

    LinearTensor final_grad(input.batch_size(), linear_layer_out);
    final_grad.setConstant(1);

    linear.forward(input.get_output());
    linear.backward(input.get_output(), final_grad);

    // expected
    MatrixXf Y, dLdX, dLdW;

    Y.resize(batch_size, linear_layer_out);
    dLdX.resize(batch_size, linear_layer_in);
    dLdW.resize(linear_layer_in, linear_layer_out);

    Y << 10, 10, 10,
      26, 26, 26;

    dLdX << 3, 3, 3, 3,
      3, 3, 3, 3;

    dLdW << 6, 6, 6,
      8, 8, 8,
      10, 10, 10,
      12, 12, 12;

    EXPECT_EQ(true, Y.isApprox(Tensor_to_Matrix(from_any<float, 2>(linear.get_output())))) << "Failed: output test";
    EXPECT_EQ(true, dLdX.isApprox(Tensor_to_Matrix(linear.get_loss_by_input_derivative()))) << "Failed dL/dX";
    EXPECT_EQ(true, dLdW.isApprox(Tensor_to_Matrix(linear.get_loss_by_weights_derivative()))) << "Failed dL/dW";
  }
}