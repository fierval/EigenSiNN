#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include "cifar10_reader.hpp"
#include <vector>
#include <tuple>
#include <algorithm>
#include <chrono>

using namespace Eigen;

typedef std::vector<Tensor<float, 3>> ImageContainer;
typedef std::vector<uint8_t> LabelContainer;

inline cifar::CIFAR10_dataset<std::vector, Tensor<float, 3>, uint8_t> read_cifar_dataset() {

  auto dataset = cifar::read_dataset_3d<std::vector, Tensor<float, 3>, uint8_t>();
  return dataset;
}

// convert loss class into categorical representation and convert to the network data type
template<typename Loss, typename Scalar>
inline Tensor<Scalar, 2> create_2d_label_tensor(std::vector<Loss>& labels, int start, int batch_size, Index n_categories) {

  array<Index, 2> dims{ (Index) batch_size, n_categories };

  Tensor<Scalar, 2> out(dims);
  out.setZero();

  Index i = batch_size * start;
  for (int j = 0; j < batch_size; j++) {
    out(j, labels[i+j]) = 1;
    j++;
  }

  return out;

}

template<typename Loss>
inline Tensor<Loss, 1> create_1d_label_tensor(std::vector<Loss>& labels) {

  Tensor<Loss, 1> out = TensorMap<Tensor<Loss, 1>>(labels.data(), labels.size());

  return out;

}

template<typename Scalar, Index Rank> 
inline Tensor<Scalar, Rank + 1> create_batch_tensor(std::vector<Tensor<Scalar, Rank>>& images, int start, int batch_size) {

  array<Index, Rank + 1> out_dims;
  for (Index i = 1; i <= Rank; i++) {
    out_dims[i] = images[0].dimension(i - 1);
  }

  out_dims[0] = batch_size;

  Tensor<Scalar, Rank + 1> output(out_dims);
  for (Index i = 0; i < batch_size; i++) {
    output.chip(i, 0) = images[i + start * batch_size];
  }

  return output;
}

template <typename Scalar, typename Loss>
inline void shuffle(ImageContainer& images, LabelContainer& labels) {
  static bool inited(false);
  static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine rand_gen;

  if (!inited) {
    inited = true;
    rand_gen = std::default_random_engine(seed);
  }
  std::vector<int> idxs(images.size());
  std::iota(idxs.begin(), idxs.end(), 0);

  std::shuffle(idxs.begin(), idxs.end(), rand_gen);

  for (Index i = 0; i < idxs.size(); i++) {
    std::iter_swap(images.begin() + i, images.begin() + idxs[i]);
    std::iter_swap(labels.begin() + i, labels.begin() + idxs[i]);
  }
}