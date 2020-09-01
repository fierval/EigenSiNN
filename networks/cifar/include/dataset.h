#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include "cifar10_reader.hpp"
#include <vector>
#include <tuple>

using namespace Eigen;

typedef std::vector<Tensor<float, 3>> ImageContainer;
typedef std::vector<uint8_t> LabelContainer;


std::tuple<ImageContainer, LabelContainer> next_batch(ImageContainer& data, LabelContainer& labels, size_t batch_size, bool restart);

cifar::CIFAR10_dataset<std::vector, Tensor<float, 3>, uint8_t> read_cifar_dataset();

// convert loss class into categorical representation and convert to the network data type
template<typename Loss>
inline Tensor<Loss, 2> create_2d_label_tensor(std::vector<Loss>& labels, Index n_categories) {

  array<Index, 2> dims{ (Index)labels.size(), n_categories };

  Tensor<Loss, 2> out(dims);
  out.setZero();

  Index i = 0;
  for (const auto& label : labels) {
    out(i, label) = 1;
    i++;
  }

  return out;

}

template<typename Loss>
inline Tensor<Loss, 1> create_1d_label_tensor(std::vector<Loss>& labels) {

  Tensor<Loss, 1> out = TensorMap<Tensor<Loss, 1>>(labels.data(), labels.size());

  return out;

}

template<typename Scalar, Index Rank> 
inline Tensor<Scalar, Rank + 1> create_batch_tensor(std::vector<Tensor<Scalar, Rank>>& images) {

  array<Index, Rank + 1> out_dims;
  for (Index i = 1; i <= Rank; i++) {
    out_dims[i] = images[0].dimension(i - 1);
  }

  out_dims[0] = images.size();

  Tensor<Scalar, Rank + 1> output(out_dims);
  for (Index i = 1; i < images.size(); i++) {
    output.chip(i, 0) = images[i];
  }

  return output;
}