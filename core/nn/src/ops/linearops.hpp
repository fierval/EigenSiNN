#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

namespace EigenSinn {
  typedef Tensor<float, 2> LinearTensor;

  inline LinearTensor adjust_linear_bias(LinearTensor& layer) {

    LinearTensor ones(layer.dimension(0), 1);
    ones.setConstant(1);

    // create new tensor and set the new columnt to "1"
    LinearTensor new_layer(layer.dimension(0), layer.dimension(1) + 1);
    new_layer.chip(layer.dimension(1), 1).setConstant(1);

    // copy the rest
    array<int, 2> offsets({ 0, 0 });
    array<int, 2> extents({ (int)layer.dimension(0), (int)layer.dimension(1) });
    new_layer.slice(offsets, extents) = layer;

    return new_layer;
  }
}