#include <iostream>
#include <gtest/gtest.h>
#include <layers/dropout.hpp>
#include "include/commondata2d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class Dropout : public ::testing::Test {

  protected:

    void SetUp() override {
      cd.init();
    }

    CommonData2d cd;
  };

  TEST_F(Dropout, Forward) {

    EigenSinn::Dropout<float, 2> dropout;
    dropout.forward(cd.linearInput);
    auto output = from_any<float, 2>(dropout.get_output());

    Tensor<float, 2> zeros(output.dimensions());
    zeros.setZero();
    Tensor<bool, 0> anyzeros = (output == zeros).any();
    EXPECT_TRUE(anyzeros(0));
  }
}