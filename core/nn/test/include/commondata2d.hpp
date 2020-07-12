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

      linearInput.setValues ({ { 0.87322980, -1.48464823, 0.63184929, 0.38559973, 0.41274598,
        -0.70019668, 0.10707247, 2.51629639},
        {-1.58931327, 0.81226659, -1.80089319, 2.07474923, -0.18125945,
        1.04950535, -0.04078181, -0.32585117},
        {-0.23254025, -1.10632432, -1.68039930, -0.50875676, -0.70717221,
        0.24772950, -0.08923696, 0.60365528} });
    }

    const array<Index, 2> dims = { 3, 8 };
    Tensor<float, 2> linearInput, linearLoss;
  };
}