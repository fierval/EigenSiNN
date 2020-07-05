#include <gtest/gtest.h>
#include <layers/input.hpp>
#include <layers/batchnorm.hpp>
#include <iostream>
#include <ops\conversions.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  TEST(Batchnorm, Forward1d) {

    const float eps = 1e-5, momentum = 0.9;

    MatrixXf input_matrix;
    input_matrix.resize(2, 5);

    input_matrix << 0.5628, 0.9343, 0.2593, 0.7921, 0.1589,
      0.1851, 0.0431, 0.5097, 0.8821, 0.5831;

    LinearTensor input = Matrix_to_Tensor(input_matrix, 2, 5);

    BatchNormalizationLayer bn(eps, momentum);

    bn.init();

    bn.forward(input);

  }
}