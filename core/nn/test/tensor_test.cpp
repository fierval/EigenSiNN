#ifdef EIGEN_USE_THREADS

#include <tensor/nntensor.hpp>
#include <gtest/gtest.h>
#include <ops/comparisons.hpp>
#include <iostream>
#include <chrono>

using namespace EigenSinn;
using namespace Eigen;

namespace EigenTest {
  TEST(NnTensorTest, InitCpu) {
    
    auto start = std::chrono::high_resolution_clock::now();

    NnTensor<float, 2> b(3000, 3000);
    Tensor<float, 2> c(3000, 3000);

    c.setConstant(2);
    b.setConstant(2);
    
    b = c.sqrt();
    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Took on a single cpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000. << std::endl;

  }

  TEST(NnTensorTest, InitThread) {

    auto start = std::chrono::high_resolution_clock::now();

    NnTensor<float, 2, threadpool> b(3000, 3000);
    Tensor<float, 2> c(3000, 3000);

    c.setConstant(2);
    b.setConstant(2);

    b = c.sqrt();
    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Took on a single cpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000. << std::endl;

  }
}
#endif