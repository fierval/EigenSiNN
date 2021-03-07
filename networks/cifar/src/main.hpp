#pragma once

#include "dataset.h"

#include <losses/crossentropyloss.hpp>
#include <device/device_tensor.hpp>
#include <chrono>
#include <vector>
#include <string>
#include "cifar10_network.hpp"

using namespace EigenSinn;

template<typename Device_, int Layout = ColMajor>
inline void TestNetwork(cifar::CIFAR10_dataset<std::vector, Tensor<float, 3, Layout>, uint8_t, Layout>& dataset, Cifar10<Device_, Layout>& net, int num_classes, std::vector<cv::String>& classes) {

  std::cout << "Starting test..." << std::endl;
  DeviceTensor<float, 4, Device_, Layout> batch_tensor(create_batch_tensor(dataset.test_images, 0, dataset.test_images.size()));
  Tensor<int, 1, Layout> label_tensor = create_1d_label_tensor<uint8_t, Layout>(dataset.test_labels).cast<int>();

  net.set_input(batch_tensor);
  net.forward();

  DeviceTensor<float, 2, Device_, Layout> test_output(net.get_output());

  // extract predicted results from the output
  DeviceTensor<Tuple<Index, float>, 2, Device_, Layout> test_index_tuples(test_output.dimensions());
  test_index_tuples.view() = test_output->index_tuples();

  // flatten the output tensor and get the "winning" indices
  DeviceTensor<Tuple<Index, float>, 1, Device_, Layout> pred_res(test_output.dimension(0));
  pred_res.view() = test_index_tuples->reduce(array<Index, 1> {1}, internal::ArgMaxTupleReducer<Tuple<Index, float>>());
  Tensor< Tuple<Index, float>, 1, Layout> h_pred_res = pred_res.to_host();

  Tensor<int, 1, Layout> predictions(pred_res.dimension(0));

  Tensor<float, 1, Layout> n_correct(num_classes), n_samples(num_classes), accuracy(num_classes);
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
  Tensor<float, 0, Layout> matches = (predictions == label_tensor).cast<float>().sum();
  std::cout << "Accuracy: " << matches(0) / predictions.dimension(0) << std::endl;

  // accuracy by class
  accuracy = n_correct / n_samples;

  for (int j = 0; j < num_classes; j++) {
    std::cout << "Class: " << classes[j] << " Accuracy: " << accuracy(j) << std::endl;
  }

}