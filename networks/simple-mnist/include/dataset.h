#pragma once

#include <mnist/mnist_reader.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

using DataContainer = std::vector<std::vector<float>>;
using LabelContainer = std::vector<uint8_t>;

mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> create_mnist_dataset();
std::tuple<DataContainer, LabelContainer> next_batch(DataContainer& data, LabelContainer& labels, size_t batch_size, bool restart);

// convert image data read by MNIST into a 2d Eigen tensor
template<typename Scalar>
inline Tensor<Scalar, 2> create_2d_image_tensor(std::vector<std::vector<float>>& data) {

  array<Index, 2> dims{ (Index)data.size(), (Index)data[0].size() };
  Tensor<Scalar, 2> data_tensor(dims);

  Index i = 0;
  for (auto& v : data) {
    
    data_tensor.chip(i, 0) = TensorMap<Tensor<Scalar, 1>>(v.data(), dims[1]);
    i++;
  }

  return data_tensor;
}

// convert loss class into categorical representation and convert to the network data type
template<typename Loss, typename Scalar>
inline Tensor<Scalar, 2> create_2d_label_tensor(std::vector<Loss>& labels, Index n_categories) {

  array<Index, 2> dims{(Index)labels.size(), n_categories};

  Tensor<Scalar, 2> out(dims);
  out.setZero();

  Index i = 0;
  for(const auto& label: labels) {
    out(i, label) = 1;
    i++;
  }
  
  return out;

}