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
  std::vector<int> test_len{ 20, 40, 80, 200, 1000, static_cast<int>(dataset.test_images.size()) };

  for (int i = 0; i < test_len.size(); i++) {
    int batch_size = test_len[i];

    //DeviceTensor<float, 4, Device_, Layout> batch_tensor(create_batch_tensor(dataset.test_images, 0, dataset.test_images.size()));

    std::cout << "============================ " << batch_size << "=================================" << std::endl;
    DeviceTensor<float, 4, Device_, Layout> batch_tensor(create_batch_tensor(dataset.test_images, 0, batch_size));
    Tensor<int, 1, Layout> label_tensor = create_1d_label_tensor<uint8_t, Layout>(dataset.test_labels).template cast<int>().slice(DSizes<Index, 1>{ 0 }, DSizes<Index, 1>{ batch_size });

    net.set_input(batch_tensor);
    net.forward();

    DeviceTensor<float, 2, Device_, Layout> d_test_output(net.get_output());
    Tensor<float, 2, Layout> test_output = d_test_output.to_host();

    //std::cout << "Test output:" << std::endl << test_output << std::endl;
    //std::cout << "Labels:" << std::endl << label_tensor << std::endl << std::endl;

    // extract predicted results from the output
    Tensor<Tuple<Index, float>, 2, Layout> test_index_tuples = test_output.index_tuples();

    // flatten the output tensor and get the "winning" indices
    Tensor<Tuple<Index, float>, 1, Layout> pred_res = test_index_tuples.reduce(array<Index, 1> {1}, internal::ArgMaxTupleReducer<Tuple<Index, float>>());
    Tensor<int, 1, Layout> predictions(pred_res.dimensions());

    Tensor<float, 1, Layout> n_correct(num_classes), n_samples(num_classes), accuracy(num_classes);
    n_correct.setZero();
    n_samples.setZero();

    // recover actual index value by unrolling the col-major stored index
    for (Index i = 0; i < pred_res.dimension(0); i++) {
      predictions(i) = (pred_res(i).first - i) / pred_res.dimension(0) % num_classes;
      int label = label_tensor(i);
      if (predictions(i) == label) {
        n_correct(label)++;
      }
      n_samples(label)++;
    }

    // overall accuracy
    Tensor<float, 0, Layout> matches = (predictions == label_tensor).template cast<float>().sum();
    std::cout << "Accuracy: " << matches(0) / predictions.dimension(0) << std::endl;

    // accuracy by class
    accuracy = n_correct / n_samples;

    for (int j = 0; j < num_classes; j++) {
      std::cout << "Class: " << classes[j] << " Accuracy: " << accuracy(j) << std::endl;
    }

    std::cout << std::endl;
  }
}

template<typename Device_, int Layout = ColMajor>
inline void TestNetworkSingleBatch(cifar::CIFAR10_dataset<std::vector, Tensor<float, 3, Layout>, uint8_t, Layout>& dataset, Cifar10<Device_, Layout>& net, int num_classes, std::vector<cv::String>& classes) {

  std::cout << "Starting test..." << std::endl;

  int total_size = dataset.test_images.size();
  Tensor<float, 1, Layout> n_correct(num_classes), n_samples(num_classes), accuracy(num_classes);
  int correct_predictions = 0;

  n_correct.setZero();
  n_samples.setZero();

  for (int i = 0; i < total_size; i++) {

    DeviceTensor<float, 4, Device_, Layout> batch_tensor(create_batch_tensor(dataset.test_images, i, 1));
    Tensor<int, 1, Layout> label_tensor = create_1d_label_tensor<uint8_t, Layout>(dataset.test_labels).template cast<int>().slice(DSizes<Index, 1>{ i }, DSizes<Index, 1>{ 1 });

    net.set_input(batch_tensor);
    net.forward();

    DeviceTensor<float, 2, Device_, Layout> d_test_output(net.get_output());
    Tensor<float, 2, Layout> test_output = d_test_output.to_host();

    // extract predicted results from the output
    Tensor<Tuple<Index, float>, 2, Layout> test_index_tuples = test_output.index_tuples();

    // flatten the output tensor and get the "winning" indices
    Tensor<Tuple<Index, float>, 1, Layout> pred_res = test_index_tuples.reduce(array<Index, 1> {1}, internal::ArgMaxTupleReducer<Tuple<Index, float>>());
    // recover actual index value by unrolling the col-major stored index
    int label = label_tensor(0);
    if (pred_res(0).first == label) {
      n_correct(label)++;
      correct_predictions++;
    }
    n_samples(label)++;
  }

  // overall accuracy
    std::cout << "Accuracy: " << static_cast<float>(correct_predictions) / total_size << std::endl;

  // accuracy by class
  accuracy = n_correct / n_samples;

  for (int j = 0; j < num_classes; j++) {
    std::cout << "Class: " << classes[j] << " Accuracy: " << accuracy(j) << std::endl;
  }

  std::cout << std::endl;

}