#pragma once

#include <gtest/gtest.h>

#include <network/cifar10_graph.hpp>

using namespace EigenSinn;

class GraphTest : public ::testing::Test {
protected:
  void SetUp() override {

  }
};

TEST_F(GraphTest, Create) {

  Cifar10<float, ThreadPoolDevice> cifar10(10);
  cifar10.print_graph();
}