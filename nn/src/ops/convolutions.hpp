#pragma once

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <utility>

#define MAX_PAD 1e6

namespace EigenSinn {
  typedef Eigen::Tensor<float, 4> ConvTensor;
  typedef Eigen::array<std::pair<int, int>, 4> Padding;

  Padding pad2dim(ConvTensor& input, int dim1, int dim2) {

    assert(dim1 >= 0 && dim1 < MAX_PAD && dim2 >= 0 && dim2 < MAX_PAD);

    Padding paddings;
    paddings[0] = std::make_pair(0, 0);
    paddings[1] = std::make_pair(dim1, dim1);
    paddings[2] = std::make_pair(dim2, dim2);
    paddings[3] = std::make_pair(0, 0);

    return paddings;
  }

  // NWHC format
  ConvTensor convolve(ConvTensor& input, ConvTensor& kernel, int pad_1 = 0, int pad_2 = 0) {
    Eigen::array<Eigen::Index, 3> dims({ 1, 2, 3 });

    assert(input.dimension(3) == kernel.dimension(3));

    // Pad if apropriate
    ConvTensor padded = input.pad(pad2dim(input, pad_1, pad_2)).eval();

    // NWHC output tensor
    ConvTensor output(padded.dimension(0), padded.dimension(1) - kernel.dimension(1) + 1, padded.dimension(2) - kernel.dimension(2) + 1, kernel.dimension(0));

    for (int i = 0; i < kernel.dimension(0); i++) {
      // final chip(0, 0) removes the third dimension
      output.chip(i, 3) = padded.convolve(kernel.chip(i, 0), dims).chip(0, 3);
    }

    return output;
  }

  ConvTensor convolve_same(ConvTensor& input, ConvTensor& kernel) {
    int dim1 = kernel.dimension(1) - 1;
    int dim2 = kernel.dimension(2) - 1;

    dim1 = dim1 / 2;
    dim2 = dim2 / 2;

    ConvTensor output = convolve(input, kernel, dim1, dim2);
    return output;
  }


  ConvTensor convolve_full(ConvTensor& input, ConvTensor& kernel) {
    int dim1 = kernel.dimension(1) - 1;
    int dim2 = kernel.dimension(2) - 1;

    ConvTensor output = convolve(input, kernel, dim1, dim2);
    return output;
  }
} // namespace EigenSinn