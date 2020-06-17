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

  Padding pad2dim(int dim1, int dim2, int dim1_1, int dim2_1) {

    assert(dim1 >= 0 && dim1 < MAX_PAD && dim2 >= 0 && dim2 < MAX_PAD && dim2_1 < MAX_PAD && dim1_1 < MAX_PAD);

    Padding paddings;
    paddings[0] = std::make_pair(0, 0);
    paddings[1] = std::make_pair(dim1, dim1_1);
    paddings[2] = std::make_pair(dim2, dim2_1);
    paddings[3] = std::make_pair(0, 0);

    return paddings;
  }

  // NWHC format
  ConvTensor convolve(ConvTensor& input, ConvTensor& kernel, Padding padding) {
    Eigen::array<Eigen::Index, 3> dims({ 1, 2, 3 });

    assert(input.dimension(3) == kernel.dimension(3));

    // Pad if apropriate
    ConvTensor padded = input.pad(padding).eval();

    // NWHC output tensor
    ConvTensor output(padded.dimension(0), padded.dimension(1) - kernel.dimension(1) + 1, padded.dimension(2) - kernel.dimension(2) + 1, kernel.dimension(0));

    for (int i = 0; i < kernel.dimension(0); i++) {
      // final chip(0, 0) removes the third dimension
      output.chip(i, 3) = padded.convolve(kernel.chip(i, 0), dims).chip(0, 3);
    }

    return output;
  }

  // pad unevenly in case of k % 2 == 1:
  // more zero's goes upfront
  ConvTensor convolve_same(ConvTensor& input, ConvTensor& kernel) {
    int dim1 = kernel.dimension(1) - 1;
    int dim2 = kernel.dimension(2) - 1;
    int dim1_1 = dim1 / 2;
    int dim2_1 = dim2 / 2;
    
    dim1 = (dim1 & 0x1) == 0 ? dim1_1 : dim1_1 + 1;
    dim2 = (dim2 & 0x1) == 0 ? dim2_1 : dim2_1 + 1;

    ConvTensor output = convolve(input, kernel, pad2dim(dim1, dim2, dim1_1, dim2_1));
    return output;
  }

  ConvTensor convolve_valid(ConvTensor& input, ConvTensor& kernel) {
    ConvTensor output = convolve(input, kernel, pad2dim(0, 0, 0, 0));
    return output;
  }


  ConvTensor convolve_full(ConvTensor& input, ConvTensor& kernel) {
    int dim1 = kernel.dimension(1) - 1;
    int dim2 = kernel.dimension(2) - 1;

    ConvTensor output = convolve(input, kernel, pad2dim(dim1, dim2, dim1, dim2));
    return output;
  }
} // namespace EigenSinn