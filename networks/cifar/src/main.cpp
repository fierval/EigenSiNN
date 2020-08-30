#include "dataset.h"

#include <layers/linear.hpp>
#include <layers/relu.hpp>
#include <layers/convolution.hpp>
#include <layers/maxpoolinglayer.hpp>
#include <losses/crossentropyloss.hpp>
#include <optimizers/adam.hpp>
#include <chrono>
#include <vector>
#include <string>
using namespace EigenSinn;

int main(int argc, char* argv[]) {

  size_t batch_size = 100;
  int num_classes = 10;
  int input_size = 28 * 28;
  int num_epochs = 2;
  float learning_rate = 0.001;
  int hidden_size = 500;

  CrossEntropyLoss<float> loss;

  std::vector<std::string> classes = { "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

  auto dataset = read_cifar_dataset();

  // get CIFAR data
 return 0;
}