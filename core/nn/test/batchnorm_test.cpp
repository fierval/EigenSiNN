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

      input_matrix << 0.5628, 0.9343, 0.2593, 0.7921, 0.1589,
        0.1851, 0.0431, 0.5097, 0.8821, 0.5831;

      input = Matrix_to_Tensor(input_matrix, batch_size, cols);

      gamma.setValues({0., 1.0, 2., 3., 4.});
      beta = gamma * 0.1f;
    }

    //void TearDown() override {}

    Tensor<float, 2> input;
    TensorSingleDim<float> beta, gamma;
    const float eps = 1e-5, momentum = 0.9;
    const Index batch_size = 2, cols = 5;
  };

  TEST_F(BatchnormTest, Forward1d) {

    BatchNormalizationLayer bn(eps, momentum);

    bn.init();

    bn.forward(input);

  }
}