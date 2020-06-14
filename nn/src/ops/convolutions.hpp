#pragma once

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <utility>

namespace EigenSinn
{
    typedef Eigen::Tensor<float, 4> ConvTensor;

    // NWHC format
    ConvTensor convolve(ConvTensor &input, ConvTensor &kernel)
    {
        Eigen::array<Eigen::Index, 3> dims({1, 2, 3});

        assert(input.dimension(3) == kernel.dimension(3));

        ConvTensor output(input.dimension(0), input.dimension(1) - kernel.dimension(1) + 1, input.dimension(2) - kernel.dimension(2) + 1, kernel.dimension(0));

        for (int i = 0; i < kernel.dimension(0); i++)
        {
            // final chip(0, 0) removes the third dimension
            output.chip(i, 3) = input.convolve(kernel.chip(i, 0), dims).chip(0, 3);
        }

        return output;
    }

    ConvTensor convolve_full(ConvTensor &input, ConvTensor &kernel)
    {
        int dim1 = kernel.dimension(1) - 1;
        int dim2 = kernel.dimension(2) - 1;

        Eigen::array<std::pair<int, int>, 4> paddings;
        paddings[0] = std::make_pair(0, 0);
        paddings[1] = std::make_pair(dim1, dim1);
        paddings[2] = std::make_pair(dim2, dim2);
        paddings[3] = std::make_pair(0, 0);

        ConvTensor padded = input.pad(paddings);

        ConvTensor output = convolve(padded, kernel);
        return output;
    }
} // namespace EigenSinn