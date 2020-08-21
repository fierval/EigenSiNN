#include "dataset.h"

#include "unsupported/Eigen/CXX11/Tensor"
#include <layers/linear.hpp>
#include <layers/relu.hpp>
#include <losses/crossentropyloss.hpp>
#include <optimizers/adam.hpp>

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
  Tensor<float, 2> label_tensor;

  auto network = create_network(batch_size, input_size, hidden_size, num_classes, learning_rate);
  init_network(network);

  CrossEntropyLoss<float> loss;

  std::vector<std::any> prev_outputs;

  do {

    // get the data
    std::tie(next_data, next_labels) = 
      next_batch(mnist_dataset.training_images, mnist_dataset.training_labels, batch_size);

    if (next_data.size() == 0) {
      break;
    }

    // convert to Eigen tensors
    data_tensor = create_2d_image_tensor<float>(next_data);
    label_tensor = create_2d_label_tensor<uint8_t, float>(next_labels, num_classes);

    // forward
    std::any tensor(data_tensor);

    for (auto it = network.begin(); it != network.end(); it++) {

      prev_outputs.push_back(tensor);
      it->layer->forward(tensor);
      tensor = it->layer->get_output();
    }

    // compute loss
    loss.forward(tensor, label_tensor);

    //backprop
    // loss gradient
    loss.backward();
    auto& back_grad = loss.get_loss_derivative_by_input();

    auto prev_out_iter = prev_outputs.rbegin();

    for (auto rit = network.rbegin(); rit != network.rend(); rit++, prev_out_iter++) {
      rit->layer->backward(*prev_out_iter, back_grad);
      back_grad = rit->layer->get_loss_by_input_derivative();
    }

    // TODO: optimizer
    prev_out_iter = prev_outputs.rbegin();
    for (auto optit = network.rbegin(); optit != network.rend(); optit++) {

    }
    
  } while(next_data.size() > 0);

  return 0;
}