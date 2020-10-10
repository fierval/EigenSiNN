#include "dataset.h"

#include "unsupported/Eigen/CXX11/Tensor"
#include <layers/linear.hpp>
#include <layers/relu.hpp>
#include <layers/input.hpp>
#include <losses/crossentropyloss.hpp>
#include <optimizers/adam.hpp>
#include <chrono>

#include "network.h"

using namespace Eigen;
using namespace EigenSinn;

int main(int argc, char* argv[]) {

  size_t batch_size = 100;
  int num_classes = 10;
  int input_size = 28 * 28;
  int num_epochs = 2;
  float learning_rate = 0.001;
  int hidden_size = 500;

  // get MNIST data
  auto mnist_dataset = create_mnist_dataset();

  DataContainer next_data;
  LabelContainer next_labels;

  // label tensor and network tensor need to have the same type
  Tensor<float, 2> data_tensor;
  Tensor<uint8_t, 2> label_tensor;

  auto network = create_network(batch_size, input_size, hidden_size, num_classes, learning_rate);
  init_network(network);

  CrossEntropyLoss<float, uint8_t, 2> loss;

  bool restart = true;

  auto start = std::chrono::high_resolution_clock::now();
  auto start_step = std::chrono::high_resolution_clock::now();
  auto stop = std::chrono::high_resolution_clock::now();

  array<Index, 2> dims{ batch_size, input_size };
  Input<float, 2> input(dims);

  // Train
  for (int i = 0; i < num_epochs; i++) {
    restart = true;
    start = std::chrono::high_resolution_clock::now();
    start_step = std::chrono::high_resolution_clock::now();
    int step = 0;

    do {

      // get the data
      std::tie(next_data, next_labels) =
        next_batch(mnist_dataset.training_images, mnist_dataset.training_labels, batch_size, restart);
      restart = false;

      if (next_data.size() == 0) {
        break;
      }

      // convert to Eigen tensors
      data_tensor = create_2d_image_tensor<float>(next_data);
      label_tensor = create_2d_label_tensor(next_labels, num_classes);

      dynamic_cast<Input<float, 2>*>(network[0].layer)->set_input(data_tensor.data());

      // forward
      for (auto it = network.begin() + 1; it != network.end(); it++) {
          it->layer->forward(*(it - 1)->layer);
      }

      // compute loss
      TensorMap<Tensor<float, 2>> output(network.rbegin()->layer->get_output(), label_tensor.dimensions());
      loss.step(output, label_tensor);

      //backprop
      // loss gradient
      auto back_grad = loss.get_loss_derivative_by_input();

      for (auto rit = network.rbegin(); rit != network.rend() - 1; rit++) {

        if (rit == network.rbegin()) {
          rit->layer->backward(*(rit + 1)->layer, back_grad);
        }
        else {
          rit->layer->backward(*(rit + 1)->layer, *(rit - 1)->layer);
        }
      }

      // optimizer
      for (auto optit = network.rbegin(); optit != network.rend(); optit++) {
        if (optit->optimizer == nullptr) {
          continue;
        }

        float* weights, * bias;
        auto layer = optit->layer;

        std::tie(weights, bias) = optit->optimizer->step(*(optit->layer));
      }

      step++;
      if (step % 100 == 0) {
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Epoch: " << i + 1 << ". Step: " << step << ". Loss: " << std::any_cast<float>(loss.get_output()) << ". Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start_step).count() / 1000. << "." << std::endl;
        start_step = std::chrono::high_resolution_clock::now();
      }

    } while (next_data.size() > 0);

    stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.;

    std::cout << "Epoch: " << i << ". Time: " << elapsed << " sec." << std::endl;
  }

  // Test
  size_t test_size = mnist_dataset.test_images.size();

  // get data
  std::tie(next_data, next_labels) =
    next_batch(mnist_dataset.test_images, mnist_dataset.test_labels, test_size, true);

  // convert to tensors
  data_tensor = create_2d_image_tensor<float>(next_data);
  Tensor<uint8_t, 1> test_label_tensor = create_1d_label_tensor<uint8_t>(next_labels);

  // forward
  std::vector<Index> test_dims{ (Index)test_size, input_size };
  network[0].layer->set_dims(test_dims, test_dims);

  dynamic_cast<Input<float, 2>*>(network[0].layer)->set_input(data_tensor.data());

  for (auto it = network.begin() + 1; it != network.end(); it++) {
      it->layer->forward(*(it - 1)->layer);
  }

  TensorMap<Tensor<float, 2>> test_output(network.rbegin()->layer->get_output(), vector2array<2>(network.rbegin()->layer->get_out_dims()));
  Tensor<Tuple<Index, float>, 2> test_index_tuples = test_output.index_tuples();
  Tensor<Tuple<Index, float>, 1> pred_res = test_index_tuples.reduce(array<Index, 1> {1}, internal::ArgMaxTupleReducer<Tuple<Index, float>>());
  Tensor<Index, 1> predictions(pred_res.dimension(0));

  for (Index i = 0; i < pred_res.dimension(0); i++) {
    predictions(i) = (pred_res(i).first - i) / pred_res.dimension(0) % num_classes;
  }

  Tensor<float, 0> matches = (predictions == test_label_tensor).cast<float>().sum();

  std::cout << "Accuracy: " << matches(0) / test_output.dimension(0) << std::endl;
  return 0;
}