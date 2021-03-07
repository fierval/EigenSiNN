#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include "cifar10_reader.hpp"
#include <vector>
#include <tuple>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


using namespace Eigen;

template<int Layout = ColMajor>
using ImageContainer =  std::vector<Tensor<float, 3, Layout>>;

typedef std::vector<uint8_t> LabelContainer;

template<int Layout = ColMajor>
inline cifar::CIFAR10_dataset<std::vector, Tensor<float, 3, Layout>, uint8_t, Layout> read_cifar_dataset() {

  auto dataset = cifar::read_dataset_3d<std::vector, Tensor<float, 3, Layout>, uint8_t, Layout>();
  return dataset;
}

// convert loss class into categorical representation and convert to the network data type
template<typename Loss, int Layout = ColMajor>
inline Tensor<Loss, 2, Layout> create_2d_label_tensor(std::vector<Loss>& labels, int start, int batch_size, Index n_categories) {

  array<Index, 2> dims{ (Index)batch_size, n_categories };

  Tensor<Loss, 2, Layout> out(dims);
  out.setZero();

  Index i = batch_size * start;
  for (int j = 0; j < batch_size; j++) {
    out(j, labels[i + j]) = 1;
  }

  return std::move(out);

}

template<typename Loss, int Layout = ColMajor>
inline Tensor<Loss, 1, Layout> create_1d_label_tensor(std::vector<Loss>& labels) {

  Tensor<Loss, 1, Layout> out = TensorMap<Tensor<Loss, 1, Layout>>(labels.data(), labels.size());

  return out;

}

template<typename Scalar, Index Rank, int Layout = ColMajor>
inline Tensor<Scalar, Rank + 1, Layout> create_batch_tensor(std::vector<Tensor<Scalar, Rank, Layout>>& images, int start, int batch_size) {

  array<Index, Rank + 1> out_dims;
  for (Index i = 1; i <= Rank; i++) {
    out_dims[i] = images[0].dimension(i - 1);
  }

  out_dims[0] = batch_size;

  Tensor<Scalar, Rank + 1, Layout> output(out_dims);
  for (Index i = 0; i < batch_size; i++) {
    output.chip(i, 0) = images[i + start * batch_size];
  }

  return std::move(output);
}

template <typename Scalar, typename Loss, int Layout = ColMajor>
inline void shuffle(ImageContainer<Layout>& images, LabelContainer& labels, bool should_shuffle = true) {

  if (!should_shuffle) { return; }

  static bool inited(false);
  static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine rand_gen;

  if (!inited) {
    inited = true;
    rand_gen = std::default_random_engine(seed);
  }
  std::vector<int> idxs(images.size()), range(images.size());
  std::iota(idxs.begin(), idxs.end(), 0);
  std::iota(range.begin(), range.end(), 0);

  std::shuffle(idxs.begin(), idxs.end(), rand_gen);
  std::vector<Tensor<float, 3, Layout>> tmp_ims(images.size());
  std::vector<uint8_t> tmp_labs(labels.size());

  std::copy(std::execution::par, images.begin(), images.end(), tmp_ims.begin());
  std::copy(std::execution::par, labels.begin(), labels.end(), tmp_labs.begin());

  std::for_each(std::execution::par, range.begin(), range.end(), [&](int n) {
    images[idxs[n]] = tmp_ims[n];
    labels[idxs[n]] = tmp_labs[n];
    });

}

// explore the dataset
template<int Layout = ColMajor>
inline void explore(cifar::CIFAR10_dataset<std::vector, Tensor<float, 3, Layout>, uint8_t, Layout>& dataset, bool should_explore = false) {

  if (!should_explore) {
    return;
  };

  // show image
  cv::Mat mat, resized, clr;
  static std::vector<std::string> classes = { "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

  for (int i = 0; i < dataset.training_images.size(); i++) {

    Tensor<float, 3, Layout> im = 255. * cifar::normalize<float>(dataset.training_images[i], false).shuffle(array<Index, 3>{1, 2, 0});

    Tensor<uint8_t, 3, Layout> im8 = im.template cast<uint8_t>();

    // TODO: This came from 4.4.0
    cv::eigen2cv(im8, mat);

    cv::cvtColor(mat, clr, cv::COLOR_BGR2RGB);
    cv::resize(clr, resized, cv::Size(120, 120));

    cv::imshow("Sample", resized);
    std::cout << "Class: " << classes[dataset.training_labels[i]] << std::endl;

    if (cv::waitKey() == 27) {
      cv::destroyAllWindows();
      break;
    }
  }
}