#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include "cifar10_reader.hpp"
#include <vector>
#include <tuple>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// copy-paste from OpenCV 4.4.0. vcpkg has 4.3.0
namespace cv {
  template <typename _Tp, int _layout> static inline
    void eigen2cv(const Eigen::Tensor<_Tp, 3, _layout>& src, OutputArray dst) {
    if (!(_layout & Eigen::RowMajorBit)) {
      const std::array<int, 3> shuffle{ 2, 1, 0 };
      Eigen::Tensor<_Tp, 3, !_layout> row_major_tensor = src.swap_layout().shuffle(shuffle);
      Mat _src(src.dimension(0), src.dimension(1), CV_MAKETYPE(DataType<_Tp>::type, src.dimension(2)), row_major_tensor.data());
      _src.copyTo(dst);
    } else {
      Mat _src(src.dimension(0), src.dimension(1), CV_MAKETYPE(DataType<_Tp>::type, src.dimension(2)), (void*)src.data());
      _src.copyTo(dst);
    }
  }
}

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

  array<Index, 2> dims{ (Index)batch_size, n_categories };

  Tensor<Scalar, 2> out(dims);
  out.setZero();

  Index i = batch_size * start;
  for (int j = 0; j < batch_size; j++) {
    out(j, labels[i + j]) = 1;
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
  std::vector<int> idxs(images.size()), range(images.size());
  std::iota(idxs.begin(), idxs.end(), 0);
  std::iota(range.begin(), range.end(), 0);

  std::shuffle(idxs.begin(), idxs.end(), rand_gen);
  std::vector<Tensor<float, 3>> tmp_ims(images.size());
  std::vector<uint8_t> tmp_labs(labels.size());

  std::copy(std::execution::par, images.begin(), images.end(), tmp_ims.begin());
  std::copy(std::execution::par, labels.begin(), labels.end(), tmp_labs.begin());

  std::for_each(std::execution::par, range.begin(), range.end(), [&](int n) {
    images[idxs[n]] = tmp_ims[n];
    labels[idxs[n]] = tmp_labs[n];
    });

}

// explore the dataset
inline void explore(cifar::CIFAR10_dataset<std::vector, Tensor<float, 3>, uint8_t>& dataset, bool skip = true) {

  // show image
  cv::Mat mat, resized, clr;
  static bool init(false);
  static std::vector<std::string> classes = { "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

  
  for (int i = 0; i < dataset.training_images.size(); i++) {

    Tensor<float, 3> im = 255. * cifar::normalize<float>(dataset.training_images[i], false).shuffle(array<Index, 3>{1, 2, 0});

    Tensor<uint8_t, 3> im8 = im.cast<uint8_t>();

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