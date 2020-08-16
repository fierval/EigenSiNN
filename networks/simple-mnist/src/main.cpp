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

  Tensor<float, 2> data_tensor;
  Tensor<uint8_t, 1> label_tensor;

  do {

    // get the data
    std::tie(next_data, next_labels) = 
      next_batch(mnist_dataset.training_images, mnist_dataset.training_labels, batch_size);

    data_tensor = create_2d_tensor<float>(next_data);
    label_tensor = create_label_tensor<uint8_t>(next_labels);

  } while(next_data.size() > 0);

  return 0;
}