#pragma once

#include <gtest/gtest.h>
#include <boost/config.hpp>

#include <algorithm>
#include <vector>
#include <utility>
#include <iostream>

#include <boost/graph/visitors.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_utility.hpp>

using namespace boost;

TEST(Boost, Create) {

  typedef boost::adjacency_list< boost::mapS, boost::vecS,
    boost::bidirectionalS,
    boost::property< boost::vertex_color_t, boost::default_color_type,
    boost::property< boost::vertex_degree_t, int,
    boost::property< boost::vertex_in_degree_t, int,
    boost::property< boost::vertex_out_degree_t, int > > > > >
    Graph;

  Graph G(5);

}