#include "dataset.h"

#include <losses/crossentropyloss.hpp>
#include <optimizers/adam.hpp>
#include <chrono>
#include <vector>
#include <string>
#include "network.h"

using namespace EigenSinn;
using namespace Eigen;

int main(int argc, char* argv[]) {

  size_t batch_size = 4;
  int num_epochs = 4;
  float learning_rate = 0.001;
  
  CrossEntropyLoss<float> loss;

  std::vector<std::string> classes = { "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
  int num_classes = classes.size();

  auto dataset = read_cifar_dataset();

  ImageContainer next_images;
  LabelContainer next_labels;

  auto network = create_network(batch_size, num_classes, learning_rate);
  init(network);

  auto start = std::chrono::high_resolution_clock::now();
  auto start_step = std::chrono::high_resolution_clock::now();
  auto stop = std::chrono::high_resolution_clock::now();

  // Train
  for (int i = 0; i < num_epochs; i++) {

    start = std::chrono::high_resolution_clock::now();
    start_step = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= dataset.test_images.size(); step++) {

      std::any batch_tensor = std::any(create_batch_tensor(dataset.training_images, step, batch_size));
      Tensor<float, 2> label_tensor = create_2d_label_tensor<uint8_t, float>(dataset.training_labels, step, batch_size, num_classes);

      // forward
      auto tensor = forward(network, batch_tensor);

      // loss
      loss.forward(tensor, label_tensor);
      loss.backward();

      // backward
      backward(network, loss.get_loss_derivative_by_input(), batch_tensor);

      // optimizer
      optimizer(network);

      if (step % 100 == 0) {
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Epoch: " << i + 1 
          << ". Step: " << step 
          << ". Loss: " << std::any_cast<float>(loss.get_output()) 
          << ". Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start_step).count() / 1000. 
          << "." << std::endl;

        start_step = std::chrono::high_resolution_clock::now();
      }

    } 

    stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.;

    std::cout << "Epoch: " << i << ". Time: " << elapsed << " sec." << std::endl;
  } 
  // get CIFAR data
  return 0;
}