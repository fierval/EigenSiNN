#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include  <iostream>
#include <boost/serialization/map.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace EigenSinnTest {

  class Serialization : public ::testing::Test {

    void SetUp() override {

    }
  };

  TEST_F(Serialization, Simple) {
    std::map<int, int> map = { {1,2}, {2,1} };
    std::stringstream ss;
    boost::archive::binary_oarchive oarch(ss);
    oarch << map;
    std::map<int, int> new_map;
    boost::archive::binary_iarchive iarch(ss);
    iarch >> new_map;
    
    EXPECT_EQ(new_map[1], map[1]);
  }
}