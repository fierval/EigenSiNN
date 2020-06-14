#pragma once

#include <Eigen/Dense>
#include "unsupported/Eigen/CXX11/Tensor"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <utility>

typedef Eigen::Tensor<float, 3> ConvTensor;
typedef Eigen::Tensor<float, 4> ConvKernel;

// NWHC format
ConvTensor convolve(ConvTensor& input, ConvKernel& kernel) {

    Eigen::array<Eigen::Index, 3> dims({0, 1, 2});

    assert(input.dimension(2) == kernel.dimension(3));

    ConvTensor output(input.dimension(0) - kernel.dimension(1) + 1, input.dimension(1) - kernel.dimension(2) + 1, kernel.dimension(0));

    for(int i = 0; i < kernel.dimension(0); i++) {

        // final chip(0, 0) removes the third dimension
        Eigen::Tensor<float, 3> cur_ker(kernel.dimension(1), kernel.dimension(2), kernel.dimension(3));
        Eigen::Tensor<float, 3> conv_res(input.dimension(0) - kernel.dimension(1) + 1, input.dimension(1) - kernel.dimension(2) + 1, 1);

        cur_ker = kernel.chip(i, 0);
        conv_res = input.convolve(cur_ker, dims);
        output.chip(i, 2) = input.convolve(cur_ker, dims).chip(0, 2);
    }

    return output;
}

ConvTensor convolve_full(ConvTensor& input, ConvKernel& kernel) {
    int dim1 = input.dimension(0) + kernel.dimension(1) - 1;
    int dim2 = input.dimension(1) + kernel.dimension(2) - 1;

    Eigen::array<std::pair<int, int>, 3> paddings;
    paddings[0] = std::make_pair(dim1, dim1);
    paddings[1] = std::make_pair(dim2, dim2);
    paddings[2] = std::make_pair(0, 0);

    ConvTensor padded = input.pad(paddings);

    ConvTensor output = convolve(padded, kernel);
    return output;
}

TEST(Convolution, SingleStride) {
    ConvTensor input(4, 4, 2);
    ConvKernel kernel(3, 3, 3, 2);

    input.setConstant(3);
    kernel.setConstant(1);

    ConvTensor output = convolve(input, kernel);
    EXPECT_EQ(output(0,0,0), 3 * 9 * 2) << "Failed value test";
    EXPECT_EQ(output.dimension(0), 2) << "Failed dim[0] test";
    EXPECT_EQ(output.dimension(1), 2) << "Failed dim[1] test";
    EXPECT_EQ(output.dimension(2), 3) << "Failed dim[2] test";
}

TEST(Convolution, Full) {
    ConvTensor input(4, 4, 2);
    ConvKernel kernel(3, 3, 3, 2);

    input.setConstant(3);
    kernel.setConstant(1);

    ConvTensor output = convolve_full(input, kernel);
    EXPECT_EQ(output(0,0,0), 3) << "Failed value test";
    EXPECT_EQ(output.dimension(0), 6) << "Failed dim[0] test";
    EXPECT_EQ(output.dimension(1), 6) << "Failed dim[1] test";
    EXPECT_EQ(output.dimension(2), 3) << "Failed dim[2] test";
}
