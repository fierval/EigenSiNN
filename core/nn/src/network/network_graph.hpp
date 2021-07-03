#pragma once

#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/named_function_params.hpp>

#include <layers/layer_base.hpp>
#include <optimizers/optimizer_base.hpp>

// the predefined property types are listed here:
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/using_adjacency_list.html#sec:adjacency-list-properties
// http://www.boost.org/doc/libs/1_76_0/libs/graph/doc/bundles.html

namespace EigenSinn {
  
  template<typename Scalar, typename Device_>
  using PtrLayer = std::shared_ptr<LayerBase<Scalar, Device_>>;

  template<typename Scalar, typename Device_>
  using PtrOptimizer = std::shared_ptr<OptimizerBase<Scalar, Device_>>;

  template<typename Scalar, typename Device_>
  struct VertexData {
    std::string op_name; //layer or operation
    PtrLayer<Scalar, Device_> layer;
    PtrOptimizer<Scalar, Device_> optimizer;

    std::pair<std::string, std::string> input_out_names;
  };

  template<typename Scalar, typename Device_>
  using NetworkGraph =  boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexData<Scalar, Device_>>;

  template<typename Scalar, typename Device_>
  class NetworkBase {

  public:
    NetworkBase() {
    }

    template<typename OptimizerType>
    std::string add(const std::string& input_name, LayerBase<Scalar, Device_>* layer) {

      return std::string();
    }

  private:
    NetworkGraph<Scalar, Device_> graph;
  };

} // namespace EigenSinn