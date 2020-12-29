#include <iostream>
#include <gtest/gtest.h>
#include "include/commondata4d.hpp"
#include "ops/comparisons.hpp"


using namespace EigenSinn;

namespace EigenSinnTest {

  class Memory : public ::testing::Test {

  protected:
    void SetUp() {
      cd.init();
    }

    CommonData4d<DefaultDevice> cd;
  };

  TEST_F(Memory, FlatColMajor) {
    Tensor<Tuple<Index, float>, 4> idx_tuples = cd.convInput->index_tuples();

    array<Index, 4> offsets{ 1, 2, 1, 3 };
    EXPECT_EQ(idx_tuples(offsets).first, to_flat_dim(cd.convInput.dimensions(), offsets));
    
  }

  TEST_F(Memory, FlatRowMajor) {
    Tensor<Tuple<Index, float>, 4, RowMajor> idx_tuples = cd.convInput->swap_layout().reshape(cd.convInput.dimensions()).index_tuples();

    array<Index, 4> offsets{ 1, 2, 1, 3 };
    Index actual = to_flat_dim<4, RowMajor>(cd.convInput.dimensions(), offsets);
    EXPECT_EQ(idx_tuples(offsets).first, actual);

  }

  TEST_F(Memory, OffsetColMajor) {
    Tensor<Tuple<Index, float>, 4> idx_tuples = cd.convInput->index_tuples();

    array<Index, 4> offsets{ 1, 2, 1, 3 };
    Index idx = idx_tuples(offsets).first;

    EXPECT_EQ(offsets, from_flat_dim(cd.convInput.dimensions(), idx));

  }

  TEST_F(Memory, OffsetRowMajor) {
    Tensor<Tuple<Index, float>, 4, RowMajor> idx_tuples = cd.convInput->swap_layout().reshape(cd.convInput.dimensions()).index_tuples();

    array<Index, 4> offsets{ 1, 2, 1, 3 };
    Index idx = idx_tuples(offsets).first;

    array<Index, 4> actual = from_flat_dim<4, RowMajor>(cd.convInput.dimensions(), idx);
    EXPECT_EQ(offsets, actual);

  }

}