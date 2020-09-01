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

  bool restart = true;

  auto start = std::chrono::high_resolution_clock::now();
  auto start_step = std::chrono::high_resolution_clock::now();
  auto stop = std::chrono::high_resolution_clock::now();

  // Train
  for (int i = 0; i < num_epochs; i++) {

    restart = true;
    start = std::chrono::high_resolution_clock::now();
    start_step = std::chrono::high_resolution_clock::now();
    int step = 0;

    do {
      std::tie(next_images, next_labels) = next_batch(dataset.training_images, dataset.training_labels, batch_size, restart);
      restart = false;

      if (next_images.size() == 0) {
        break;
      }

      Tensor<float, 4> batch_tensor = create_batch_tensor(next_images);
      Tensor<uint8_t, 2> label_tensor = create_2d_label_tensor(next_labels, num_classes);
    } while (next_images.size() > 0);
  } 
  // get CIFAR data
  return 0;
}