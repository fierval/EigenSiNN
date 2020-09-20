
#include <gtest/gtest.h>
#include "include/commondata2d.hpp"
#include <gpu/tensor.hpp>

template <typename Scalar, int Dim>
inline auto is_elementwise_approx_eq(Tensor<Scalar, Dim> a, Tensor<Scalar, Dim> b, float prec = 1e-5) {

  Tensor<Scalar, Dim> diff = a - b;
  Tensor<Scalar, 0> res = diff.abs().maximum();
  return res(0) <= prec;
}

namespace EigenSinnTest {

  class CudaTest : public ::testing::Test {

  protected:

    void SetUp() override {
      cd.init();
    }

    CommonData2d cd;
  };

  TEST_F(CudaTest, Copy) {

    Tensor<float, 2> d_input = EigenSinn::to_gpu(cd.linearInput);
    Tensor<float, 2> input = EigenSinn::from_gpu(input);

    EXPECT_TRUE(is_elementwise_approx_eq(cd.linearInput, input));
    EigenSinn::free_gpu(d_input);

  }
}