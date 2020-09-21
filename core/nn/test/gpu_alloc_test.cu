
#include <gtest/gtest.h>
#include "include/commondata2d.hpp"
#include <gpu/tensor.hpp>

template <typename Scalar, int Dim>
inline auto is_elementwise_approx_eq(Tensor<Scalar, Dim>& a, Tensor<Scalar, Dim>& b, float prec = 1e-5) {

  Tensor<Scalar, Dim> diff = a - b;
  Tensor<Scalar, 0> res = diff.abs().maximum();
  return res(0) <= prec;
}

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