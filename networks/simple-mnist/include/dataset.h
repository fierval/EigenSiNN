#pragma once

#include <mnist/mnist_reader.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

using DataContainer = std::vector<std::vector<float>>;
using LabelContainer = std::vector<uint8_t>;

mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> create_mnist_dataset();
std::tuple<DataContainer, LabelContainer> next_batch(DataContainer& data, LabelContainer& labels, size_t batch_size);

template<typename Scalar>
inline Tensor<Scalar, 2> create_2d_tensor(std::vector<std::vector<float>>& data) {

  array<Index, 2> dims{ (Index)data.size(), (Index)data[0].size() };
  Tensor<Scalar, 2> data_tensor(dims);

  Index i = 0;
  for (auto& v : data) {
    
    data_tensor.chip(i, 0) = TensorMap<Tensor<Scalar, 1>>(v.data(), dims[1]);
    i++;
  }

  return data_tensor;
}

template<typename Scalar>
inline Tensor<Scalar, 1> create_label_tensor(std::vector<Scalar>& labels) {

  Tensor<Scalar, 1> labels_tensor = TensorMap<Tensor<Scalar, 1>>(labels.data(), (Index)labels.size());
  return labels_tensor;

}