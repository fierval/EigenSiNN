#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include  <iostream>
#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace EigenSinnTest {

  class Serialization : public ::testing::Test {

    void SetUp() override {

    }
  };

  TEST_F(Serialization, Simple) {
    std::map<int, int> map = { {1,2}, {2,1} };
    std::stringstream ss;
    boost::archive::text_oarchive oarch(ss);
    oarch << map;
    std::map<int, int> new_map;
    boost::archive::text_iarchive iarch(ss);
    iarch >> new_map;
    std::cout << (map == new_map) << std::endl;
  }
}