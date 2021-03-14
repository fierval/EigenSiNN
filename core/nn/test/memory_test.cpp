#include <iostream>
#include <gtest/gtest.h>
#include "include/commondata4d.hpp"
#include "ops/comparisons.hpp"
#include "layers/linear.hpp"
#include "layers/convolution.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Memory : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
    }

    CommonData4d<ThreadPoolDevice> cd;
  };

  TEST_F(Memory, FlatColMajor) {
    Tensor<Tuple<Index, float>, 4> idx_tuples = cd.convInput->index_tuples();

    DSizes<Index, 4> offsets{ 1, 2, 1, 3 };
    EXPECT_EQ(idx_tuples(offsets).first, to_flat_dim(cd.convInput.dimensions(), offsets));
    
  }

  TEST_F(Memory, FlatRowMajor) {
    Tensor<Tuple<Index, float>, 4, RowMajor> idx_tuples = cd.convInput->swap_layout().reshape(cd.convInput.dimensions()).index_tuples();

    DSizes<Index, 4> offsets{ 1, 2, 1, 3 };
    Index actual = to_flat_dim<Index, 4, RowMajor>(cd.convInput.dimensions(), offsets);
    EXPECT_EQ(idx_tuples(offsets).first, actual);

  }

  TEST_F(Memory, OffsetColMajor) {
    Tensor<Tuple<Index, float>, 4> idx_tuples = cd.convInput->index_tuples();

    DSizes<Index, 4> offsets{ 1, 2, 1, 3 };
    Index idx = idx_tuples(offsets).first;

    EXPECT_EQ(offsets, from_flat_dim(cd.convInput.dimensions(), idx));

  }

  TEST_F(Memory, OffsetRowMajor) {
    Tensor<Tuple<Index, float>, 4, RowMajor> idx_tuples = cd.convInput->swap_layout().reshape(cd.convInput.dimensions()).index_tuples();

    DSizes<Index, 4> offsets{ 1, 2, 1, 3 };
    Index idx = idx_tuples(offsets).first;

    DSizes<Index, 4> actual = from_flat_dim<Index, 4, RowMajor>(cd.convInput.dimensions(), idx);
    EXPECT_EQ(offsets, actual);

  }

  TEST_F(Memory, AssignmentLayer) {

    Linear<float> l1(5, 7);
    Linear<float> l2 = l1;

    Conv2d<float> c1({ 5, 3, 2, 4 });
    Conv2d<float> c2 = c1;
  }

}