#include <gtest/gtest.h>
#include <ops/batchnorm.hpp>
#include <ops/convolutions.hpp>
#include <layers/batchnorm.hpp>
#include <iostream>

using namespace EigenSinn;

namespace EigenSinnTest {

  TEST(Batchnorm, Broadcast) {
    TensorSingleDim b(5);
    b.setValues({1, 2, 3, 4, 5});

    Eigen::Tensor<float, 3> a(5, 2, 3), c, d;

    a.setZero();
    c = b.reshape(Eigen::array<Index, 3>{5, 1, 1}).broadcast(Eigen::array<Index, 3>{1, 2, 3});
    d = a + c;
    std::cerr << d;
  }

  TEST(Batchnorm, Pow) {
    TensorSingleDim b(5);
    b.setValues({ 1, 2, 3, 4, 5 });

    std::cerr << b.pow(2.) << std::endl;
  }

  TEST(Batchnorm, NNTensor) {
    NnTensor<4> a(2, 3, 5, 7);

    a.setRandom();
    DimensionList<DenseIndex, 3> dims;
    std::cerr << a << std::endl;
    Tensor<float, 1> reduced = a.reduce(dims, internal::MinReducer<float>());
    std::cerr << "Reduced: " << std::endl << reduced << std::endl;
  }
}