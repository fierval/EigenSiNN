//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains functions to read the CIFAR-10 dataset
 */

#ifndef CIFAR10_READER_HPP
#define CIFAR10_READER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <execution>
#include <algorithm>
#include <numeric>

using namespace Eigen;

namespace cifar {

  /*!
   * \brief Represents a complete CIFAR10 dataset
   * \tparam Container The container to use
   * \tparam Image The type of image
   * \tparam Label The type of label
   */
  template <template <typename...> class Container, typename Image, typename Label, int Layout = RowMajor>
  struct CIFAR10_dataset {
    Container<Image> training_images; ///< The training images
    Container<Image> test_images;     ///< The test images
    Container<Label> training_labels; ///< The training labels
    Container<Label> test_labels;     ///< The test labels
  };

  // normalize_denormalize = true - normalize. Otherwise denormalize
  template<typename Scalar, int Layout>
  inline Tensor<Scalar, 3, Layout> normalize(Tensor<Scalar, 3, Layout>& raw, bool normalize_denormalize = true) {
    static bool inited(false);
    static Tensor<Scalar, 1, Layout> mean(3), std(3);
    static Tensor<Scalar, 3, Layout> broad_mean, broad_std;

    if (!inited) {
      inited = true;
      mean.setValues({ 0.5, 0.5, 0.5 });
      std.setValues({ 0.5, 0.5, 0.5 });

      //mean.setValues({ 0.4914, 0.4822, 0.4465 });
      //std.setValues({ 0.247, 0.243, 0.261 });

      broad_mean = mean.reshape(array<Index, 3>{ 3,1,1 }).broadcast(array<Index, 3>{1, 32, 32});
      broad_std = std.reshape(array<Index, 3>{ 3,1,1 }).broadcast(array<Index, 3>{1, 32, 32});
    }

    Tensor<Scalar, 3, Layout> res;
    if (normalize_denormalize) {
      res = (raw - broad_mean) / broad_std;
    }
    else {
      res = raw * broad_std + broad_mean;
    }

    return res;
  }

    /*!
   * \brief Read a CIFAR 10 data file inside the given containers
   * \param images The container to fill with the labels
   * \param path The path to the label file
   * \param limit The maximum number of elements to read (0: no limit)
   */
  template <typename Images, typename Labels, int Layout>
  void read_cifar10_file(Images& images, Labels& labels, const std::string& path, std::size_t limit) {

    if (limit && limit <= images.size()) {
      return;
    }

    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (!file) {
      std::cout << "Error opening file: " << path << std::endl;
      return;
    }

    auto file_size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[file_size]);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), file_size);
    file.close();

    std::size_t start = images.size();

    size_t size = 10000;
    size_t capacity = limit - images.size();

    if (capacity > 0 && capacity < size) {
      size = capacity;
    }
    
    std::vector<int> cifar_single_entry_range(size);
    std::iota(cifar_single_entry_range.begin(), cifar_single_entry_range.end(), 0);

    // Prepare the size for the new
    images.resize(images.size() + size);
    labels.resize(labels.size() + size);


    Tensor<float, 3, Layout> naught(3, 32, 32);
    naught.setZero();

    // so we simply initialize normalization
    // and not run into race conditions during parallel
    // foreach
    normalize(naught);

    std::for_each(std::execution::par, cifar_single_entry_range.begin(), cifar_single_entry_range.end(), [&](int i) {
      labels[start + i] = buffer[i * 3073];

      TensorMap<Tensor<char, 3, RowMajor>> im_value(&buffer[i * 3073 + 1], 3, 32, 32);

      // tensors are column-major and the image we get is row-major.
      // normalize them while at it.
#ifdef COLMAJOR
      Tensor<float, 3, Layout> col_major = 1. / 255 * im_value.swap_layout().cast<byte>().cast<float>().shuffle(array<Index, 3>{2, 1, 0});
#else
      Tensor<float, 3, Layout> col_major = 1. / 255 * im_value.cast<byte>().cast<float>();
#endif
      Tensor<float, 3, Layout> normalized = normalize(col_major);
      
      images[start + i] = normalized;

      });
  }

  /*!
   * \brief Read all test data.
   *
   * The dataset is assumed to be in a cifar-10 subfolder
   *
   * \param limit The maximum number of elements to read (0: no limit)
   * \param func The functor to create the image objects.
   */
  template <typename Images, typename Labels, int Layout>
  void read_test(const std::string& folder, std::size_t limit, Images& images, Labels& labels) {
    read_cifar10_file<Images, Labels, Layout>(images, labels, folder + "/test_batch.bin", limit);
  }

  /*!
   * \brief Read all training data
   *
   * The dataset is assumed to be in a cifar-10 subfolder
   *
   * \param limit The maximum number of elements to read (0: no limit)
   * \param func The functor to create the image objects.
   */
  template <typename Images, typename Labels, int Layout>
  void read_training(const std::string& folder, std::size_t limit, Images& images, Labels& labels) {
    read_cifar10_file<Images, Labels, Layout>(images, labels, folder + "/data_batch_1.bin", limit);
    read_cifar10_file<Images, Labels, Layout>(images, labels, folder + "/data_batch_2.bin", limit);
    read_cifar10_file<Images, Labels, Layout>(images, labels, folder + "/data_batch_3.bin", limit);
    read_cifar10_file<Images, Labels, Layout>(images, labels, folder + "/data_batch_4.bin", limit);
    read_cifar10_file<Images, Labels, Layout>(images, labels, folder + "/data_batch_5.bin", limit);
  }

  /*!
   * \brief Read all test data.
   *
   * The dataset is assumed to be in a cifar-10 subfolder
   *
   * \param limit The maximum number of elements to read (0: no limit)
   * \param func The functor to create the image objects.
   */
  template <typename Images, typename Labels, int Layout>
  void read_test(std::size_t limit, Images& images, Labels& labels) {
    read_test<Images, Labels, Layout>(CIFAR_DATA_LOCATION, limit, images, labels);
  }

  /*!
   * \brief Read all training data
   *
   * The dataset is assumed to be in a cifar-10 subfolder
   *
   * \param limit The maximum number of elements to read (0: no limit)
   * \param func The functor to create the image objects.
   */
  template <typename Images, typename Labels, int Layout>
  void read_training(std::size_t limit, Images& images, Labels& labels) {
    read_training<Images, Labels, Layout>(CIFAR_DATA_LOCATION, limit, images, labels);
  }
    
  template <template <typename...> class Container, typename Image, typename Label = uint8_t, int Layout = RowMajor>
  CIFAR10_dataset<Container, Image, Label, Layout> read_dataset_3d(std::size_t training_limit = 0, std::size_t test_limit = 0) {

    CIFAR10_dataset<Container, Image, Label, Layout> dataset;

    read_training<decltype(dataset.training_images), decltype(dataset.training_labels), Layout>(training_limit, dataset.training_images, dataset.training_labels);
    read_test<decltype(dataset.training_images), decltype(dataset.training_labels), Layout>(test_limit, dataset.test_images, dataset.test_labels);

    return dataset;
  }
} //end of namespace cifar

#endif
