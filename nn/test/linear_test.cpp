#include "input.hpp"
#include "linear.hpp"
#include "Random.h"
#include <iostream>

#include <gtest/gtest.h>

namespace nn_tests {

    TEST(FullyConnected, BackpropTest) {
        const int linear_layer_out = 3, batch_size = 2, linear_layer_in = 4;
        MatrixXd input_matrix;
        input_matrix.resize(batch_size, linear_layer_in);
        input_matrix <<
            1, 2, 3, 4,
            5, 6, 7, 8;

        //initialization constants
        RNG rng(1);
        double mu = 0;
        double sigma = 0.01;

        InputLayer input(input_matrix);
        Linear linear(input.batch_size(), input.input_vector_dim(), linear_layer_out);

        linear.init(rng, mu, sigma, true);

        MatrixXd final_grad = MatrixXd::Ones(input.batch_size(), linear_layer_out);
        linear.forward(input.get_layer());
        linear.backward(input.get_layer(), final_grad);

        // expected
        MatrixXd Y, dLdX, dLdW;

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

        EXPECT_EQ(true, Y.isApprox(linear.get_output())) << "Failed: output test";
        EXPECT_EQ(true, dLdX.isApprox(linear.get_loss_by_input_derivative())) << "Failed dL/dX";
        EXPECT_EQ(true, dLdW.isApprox(linear.get_loss_by_weights_derivative())) << "Failed dL/dW";
    }
}