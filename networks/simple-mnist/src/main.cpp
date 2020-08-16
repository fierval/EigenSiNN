#include "unsupported/Eigen/CXX11/Tensor"
#include <layers/linear.hpp>
#include <layers/relu.hpp>
#include <losses/crossentropyloss.hpp>
#include <optimizers/adam.hpp>

#include <mnist/mnist_reader.hpp>

using namespace Eigen;

int main(int argc, char* argv[]) {

  // create MNIST dataset: https://github.com/wichtounet/mnist 

  //MNIST_DATA_LOCATION set by MNIST in cmake
  std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

  return 0;
}