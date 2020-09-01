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

    for (int step = 1; step <= dataset.training_images.size() / batch_size; step++) {

      std::any batch_tensor = std::any(create_batch_tensor(dataset.training_images, step - 1, batch_size));
      Tensor<float, 2> label_tensor = create_2d_label_tensor<uint8_t, float>(dataset.training_labels, step - 1, batch_size, num_classes);

      // forward
      auto tensor = forward(network, batch_tensor);

      // loss
      loss.forward(tensor, label_tensor);
      loss.backward();

      // backward
      backward(network, loss.get_loss_derivative_by_input(), batch_tensor);

      // optimizer
      optimizer(network);

      if (step % 2000 == 0) {
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

    std::cout << "Epoch: " << i + 1 << ". Time: " << elapsed << " sec." << std::endl;
  } 
  
  // Test
  std::cout << "Starting test..." << std::endl;
  std::any batch_tensor = std::any(create_batch_tensor(dataset.test_images, 0, dataset.test_images.size()));
  auto label_tensor = create_1d_label_tensor(dataset.test_labels);

  auto tensor = forward(network, batch_tensor);

  Tensor<float, 2> test_output = from_any<float, 2>(tensor);
  Tensor<Tuple<Index, float>, 2> test_index_tuples = test_output.index_tuples();
  Tensor<Tuple<Index, float>, 1> pred_res = test_index_tuples.reduce(array<Index, 1> {1}, internal::ArgMaxTupleReducer<Tuple<Index, float>>());
  Tensor<Index, 1> predictions(pred_res.dimension(0));

  // overall accuracy
  Tensor<float, 0> matches = (predictions == label_tensor).cast<float>().sum();
  std::cout << "Accuracy: " << matches(0) / label_tensor.dimension(0) << std::endl;

  Tensor<float, 1> n_correct(num_classes), n_samples(num_classes), accuracy(num_classes);
  n_correct.setZero();
  n_samples.setZero();

  // recover actual index value by unrolling the col-major stored index
  for (Index i = 0; i < pred_res.dimension(0); i++) {
    predictions(i) = (pred_res(i).first - i) / pred_res.dimension(0) % num_classes;
    int label = (int)label_tensor(i);
    if (predictions(i) == label) {
      n_correct(label)++;
    }
    n_samples(label)++;
  }
  
  accuracy = n_correct / n_samples;

  for (int j = 0; j < num_classes; j++) {
    std::cout << "Class: " << classes[j] << " Accuracy: " << accuracy(j) << std::endl;
  }

  return 0;
}