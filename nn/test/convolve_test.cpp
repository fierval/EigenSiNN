#pragma once

#include <Eigen/Dense>
#include "unsupported/Eigen/CXX11/Tensor"
#include <cassert>

typedef Eigen::Tensor<float, 3> ConvTensor;
typedef Eigen::Tensor<float, 4> ConvKernel;

// NWHC format
ConvTensor convolve(ConvTensor& input, ConvKernel& kernel) {

    Eigen::array<Eigen::Index, 3> dims({0, 1, 2});

    assert(input.dimension(2) == kernel.dimension(3));

    ConvTensor output(input.dimension(0) - kernel.dimension(1) + 1, input.dimension(1) - kernel.dimension(2) + 1, kernel.dimension(0));

    for(int i = 0; i < kernel.dimension(0); i++) {

        // final chip(0, 0) removes the third dimension
        output.chip(i, 2) = input.convolve(kernel.chip(i, 0), dims).chip(0, 0);
    }

    return output;
}

void test_tensor_ops() {


}
