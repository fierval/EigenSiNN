#include "unsupported/Eigen/CXX11/Tensor"
#include <layers/linear.hpp>

using namespace Eigen;

int main(int argc, char* argv[]) {

  EigenSinn::Linear<float> fc(3, 2, 4);
  Tensor<float, 2> t;
  
  t.resize(2, 2);
  t.setZero();

  return 0;
}