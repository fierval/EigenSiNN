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
  int num_classes = 10;
  int num_epochs = 4;
  float learning_rate = 0.001;
  
  CrossEntropyLoss<float> loss;

  std::vector<std::string> classes = { "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

  auto dataset = read_cifar_dataset();
  auto network = create_network(batch_size, num_classes, learning_rate);

  // get CIFAR data
  return 0;
}