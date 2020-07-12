#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

using namespace EigenSinn;
using namespace Eigen;

namespace EigenSinnTest {

  struct CommonData2d {

    // data will be presented in NHWC format
    CommonData2d() {} 
    
    void init() {
      linearLoss.resize(dims);
      linearInput.resize(dims);

      linearLoss.setValues({ {0.36603546, 0.40687686, 0.87746394, 0.62148917, 0.86859787, 0.51380110,
        0.60315830, 0.00604892},
        {0.21118003, 0.13755852, 0.23697436, 0.16111487, 0.92154074, 0.52772647,
        0.00271451, 0.32741523},
        {0.35232794, 0.79406255, 0.54326051, 0.46549028, 0.16354656, 0.50212926,
        0.17194599, 0.52774191} });

      linearInput.setValues({ { -3.78266811, -0.59510386, 0.21390513, 0.48233253, -0.70359427,
        1.93920422, -0.25173056, 0.46881983 },
        {0.26570877, 0.21364078, -0.81126142, 1.88566184, 0.52365559,
        -0.88812321, -0.28724664, -0.93524176},
        {0.26916730, 0.73577052, -0.38462478, -1.14686072, 1.12185359,
        -0.72033381, -0.39438322, 1.14444125} });
    }

    const array<Index, 2> dims = { 3, 8 };
    Tensor<float, 2> linearInput, linearLoss;
  };
}