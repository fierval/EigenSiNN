#include "dataset.h"

#include "unsupported/Eigen/CXX11/Tensor"
#include <layers/linear.hpp>
#include <layers/relu.hpp>
#include <losses/crossentropyloss.hpp>
#include <optimizers/adam.hpp>

using namespace Eigen;


int main(int argc, char* argv[]) {

  size_t batch_size = 100;

  // get MNIST data
  auto mnist_dataset = create_mnist_dataset();

  DataContainer next_data;
  LabelContainer next_labels;

  do {

    // get the data
    std::tie(next_data, next_labels) = 
      next_batch(mnist_dataset.training_images, mnist_dataset.training_labels, batch_size);

  } while(next_data.size() > 0);

  return 0;
}