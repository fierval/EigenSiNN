
#include <gtest/gtest.h>
#include "include/commondata2d.hpp"
#include <ops/comparisons.hpp>
#include <gpu/tensor.hpp>

template <typename Scalar, int Dim>

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#endif
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int;

namespace EigenSinnTest {

  class CudaTest : public ::testing::Test {

  protected:

    void SetUp() override {
      cd.init();
    }

    CommonData2d cd;
  };

  TEST_F(CudaTest, Copy) {

    float * d_input = EigenSinn::to_gpu(cd.linearInput);
    TensorMap<Tensor<float, 2>> gpu_input(d_input, cd.linearInput.dimensions());
    float* input = EigenSinn::from_gpu(gpu_input.data(), gpu_input.size());

    EigenSinn::free_gpu(d_input);

  }
}