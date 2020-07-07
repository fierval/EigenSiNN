#include <gtest/gtest.h>
#include <layers/input.hpp>
#include <layers/batchnorm.hpp>
#include <iostream>
#include <ops\conversions.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class BatchnormTest : public ::testing::Test {
  protected:
    void SetUp() override {

      MatrixXf input_matrix;
      input_matrix.resize(batch_size, cols);

      input_matrix << 0.56279999, 0.93430001, 0.25929999, 0.79210001, 0.15889999,
        0.18510000, 0.04310000, 0.50970000, 0.88209999, 0.58310002;

      input = Matrix_to_Tensor(input_matrix, batch_size, cols);
      gamma.resize(cols);
      beta.resize(cols);

      gamma.setValues({1., 2.0, 3., 4., 5.});
      beta = gamma * 0.1f;
    }

    //void TearDown() override {}

    Tensor<float, 2> input;
    TensorSingleDim<float> beta, gamma;
    const float eps = 1e-5, momentum = 0.9;
    const Index batch_size = 2, cols = 5;
  };

  TEST_F(BatchnormTest, Forward1d) {

    BatchNormalizationLayer bn(cols, eps, momentum);

    MatrixXf expected(batch_size, cols); 
    
    expected << 1.09985983, 2.19994974, -2.69904351, -3.59016228, -4.49944401,
      -0.89985991, -1.79994965, 3.29904366, 4.39015722, 5.49944448;

    bn.init(beta, gamma);
    bn.forward(input);

    EXPECT_TRUE(expected.isApprox(Tensor_to_Matrix(bn.get_output()), 1e-6));
  }
}