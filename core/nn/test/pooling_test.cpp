#include <iostream>
#include <gtest/gtest.h>
#include <layers/poolinglayer.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {
  TEST(Pool, Validate4d) {

    NnTensor<4> t(1, 4, 4, 3);
    t.setConstant(1);

    auto dims = t.dimensions();
    array<int, 2> extents({ 2, 2 });
    int stride = 2;

    bool res = check_valid_params<4>(extents, stride, dims);

    EXPECT_TRUE(res);
  }

  TEST(Pool, Validate2d) {

    NnTensor<2> t(4, 4);
    t.setConstant(1);

    auto dims = t.dimensions();
    array<int, 1> extents({ 2 });
    int stride = 2;

    bool res = check_valid_params<2>(extents, stride, dims);

    EXPECT_TRUE(res);
  }

  TEST(Pool, BadExtent) {

    NnTensor<2> t(4, 4);
    t.setConstant(1);

    auto dims = t.dimensions();
    array<int, 1> extents({ 5 });
    int stride = 2;

    bool res = check_valid_params<2>(extents, stride, dims);

    EXPECT_FALSE(res);
  }

  TEST(Pool, BadStride4d) {

    NnTensor<4> t(3, 10, 10, 4);
    t.setConstant(1);

    auto dims = t.dimensions();
    array<int, 2> extents({ 3, 3 });
    int stride = 2;

    bool res = check_valid_params<4>(extents, stride, dims);

    EXPECT_FALSE(res);
  }

}