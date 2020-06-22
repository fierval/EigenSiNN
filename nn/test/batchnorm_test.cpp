#include <gtest/gtest.h>
#include <ops/batchnorm.hpp>
#include <ops/convolutions.hpp>

#include <iostream>

using namespace EigenSinn;

namespace EigenSinnTest {

  TEST(Batchnorm, Broadcast) {
    ConvTensorSingle b(5);
    b.setValues({1, 2, 3, 4, 5});

    Eigen::Tensor<float, 3> a(5, 2, 3), c, d;

    a.setZero();
    c = b.reshape(Eigen::array<Index, 3>{5, 1, 1}).broadcast(Eigen::array<Index, 3>{1, 2, 3});
    d = a + c;
    std::cerr << d;
  }
}