#include <tensor/nntensor.hpp>
#include <gtest/gtest.h>
#include <ops/comparisons.hpp>
#include <iostream>

using namespace EigenSinn;
using namespace Eigen;

namespace EigenTest {
  TEST(NnTensorTest, Init) {
    
    Tensor<float, 2> a(3, 3);
    Tensor<float, 2> b(3, 3);
    b.setConstant(2);
    
    a = b.sqrt();
    b = b.sqrt();
    
    std::cout << a << std::endl;
    std::cout << b << std::endl;

    Tensor<float, 2> c(3, 3);
    c.setConstant(2);
    c = c.sqrt();
    std::cout << c << std::endl;

    EXPECT_TRUE(is_elementwise_approx_eq(a, b));
  }

}