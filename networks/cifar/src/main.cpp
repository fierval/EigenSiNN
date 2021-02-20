#include "dataset.h"

#include <losses/crossentropyloss.hpp>
#include <device/device_tensor.hpp>
#include <chrono>
#include <vector>
#include <string>
#include "network.h"

using namespace EigenSinn;
using namespace Eigen;

bool should_shuffle = true;
bool explore_dataset = false;
bool debug_init = false;

int main(int argc, char* argv[]) {

  size_t batch_size = 10;
  int side = 32;
  int num_epochs = 4;
  int channels = 3;
  float learning_rate = 0.001;
  
  CrossEntropyLoss<float, uint8_t, 2, ColMajor, ThreadPoolDevice> loss;

  std::vector<std::string> classes = { "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
  int num_classes = classes.size();

  auto dataset = read_cifar_dataset();

  explore(dataset, explore_dataset);

  ImageContainer next_images;
  LabelContainer next_labels;

  auto network = create_network(array<Index, 4>{(Index)batch_size, channels, side, side}, num_classes, learning_rate);
  init(network, debug_init);

  auto start = std::chrono::high_resolution_clock::now();
  auto start_step = std::chrono::high_resolution_clock::now();
  auto stop = std::chrono::high_resolution_clock::now();

  // Train
  for (int i = 0; i < num_epochs; i++) {

    start = std::chrono::high_resolution_clock::now();
    start_step = std::chrono::high_resolution_clock::now();
    
    shuffle<float, uint8_t>(dataset.training_images, dataset.training_labels, should_shuffle);

    for (int step = 1; step <= dataset.training_images.size() / batch_size; step++) {

      auto batch_tensor = DeviceTensor<ThreadPoolDevice, float, 4>(create_batch_tensor(dataset.training_images, step - 1, batch_size));
      auto label_tensor = DeviceTensor<ThreadPoolDevice, uint8_t, 2>(create_2d_label_tensor<uint8_t>(dataset.training_labels, step - 1, batch_size, num_classes));

      // forward
      dynamic_cast<Input<float, 4, ColMajor, ThreadPoolDevice>*>(network[0].layer)->set_input(batch_tensor);

      forward(network);
      assert(_CrtCheckMemory());

      // loss
      DeviceTensor<ThreadPoolDevice, float, 2> output(network.rbegin()->layer->get_output());
      loss.step(output, label_tensor);

      // std::cout << "Step: " << step << ". Loss: " << std::any_cast<float>(loss.get_output()) << std::endl;

      // backward
      backward(network, loss.get_loss_derivative_by_input());
      assert(_CrtCheckMemory());

      // optimizer
      optimizer(network);
      assert(_CrtCheckMemory());

      if (step % 1000 == 0) {
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
  DeviceTensor<ThreadPoolDevice, float, 4> batch_tensor(create_batch_tensor(dataset.test_images, 0, dataset.test_images.size()));
  Tensor<int, 1> label_tensor = create_1d_label_tensor(dataset.test_labels).cast<int>();

  auto inp_layer = dynamic_cast<Input<float, 4, ColMajor, ThreadPoolDevice>*>(network[0].layer);
  inp_layer->set_input(batch_tensor);

  forward(network);

  DeviceTensor<ThreadPoolDevice, float, 2> test_output(network.rbegin()->layer->get_output());
  DeviceTensor<ThreadPoolDevice, Tuple<Index, float>, 2> test_index_tuples(test_output.dimensions());
  test_index_tuples.view() = test_output->index_tuples();

  DeviceTensor<ThreadPoolDevice, Tuple<Index, float>, 1> pred_res(test_output.dimension(0));
  pred_res.view() = test_index_tuples->reduce(array<Index, 1> {1}, internal::ArgMaxTupleReducer<Tuple<Index, float>>());
  Tensor< Tuple<Index, float>, 1> h_pred_res = pred_res.to_host();

  Tensor<int, 1> predictions(pred_res.dimension(0));

  Tensor<float, 1> n_correct(num_classes), n_samples(num_classes), accuracy(num_classes);
  n_correct.setZero();
  n_samples.setZero();

  // recover actual index value by unrolling the col-major stored index
  for (Index i = 0; i < pred_res.dimension(0); i++) {
    predictions(i) = (h_pred_res(i).first - i) / pred_res.dimension(0) % num_classes;
    int label = label_tensor(i);
    if (predictions(i) == label) {
      n_correct(label)++;
    }
    n_samples(label)++;
  }
  
  // overall accuracy
  Tensor<float, 0> matches = (predictions == label_tensor).cast<float>().sum();
  std::cout << "Accuracy: " << matches(0) / predictions.dimension(0) << std::endl;

  // accuracy by class
  accuracy = n_correct / n_samples;

  for (int j = 0; j < num_classes; j++) {
    std::cout << "Class: " << classes[j] << " Accuracy: " << accuracy(j) << std::endl;
  }
  
  return 0;
}