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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <execution>
#include <algorithm>
#include <numeric>


namespace cifar {

  /*!
   * \brief Represents a complete CIFAR10 dataset
   * \tparam Container The container to use
   * \tparam Image The type of image
   * \tparam Label The type of label
   */
  template <template <typename...> class Container, typename Image, typename Label>
  struct CIFAR10_dataset {
    Container<Image> training_images; ///< The training images
    Container<Image> test_images;     ///< The test images
    Container<Label> training_labels; ///< The training labels
    Container<Label> test_labels;     ///< The test labels

    /*!
     * \brief Resize the training set to new_size
     *
     * If new_size is less than the current size, this function has no effect.
     *
     * \param new_size The size to resize the training sets to.
     */
    void resize_training(std::size_t new_size) {
      if (training_images.size() > new_size) {
        training_images.resize(new_size);
        training_labels.resize(new_size);
      }
    }

    /*!
     * \brief Resize the test set to new_size
     *
     * If new_size is less than the current size, this function has no effect.
     *
     * \param new_size The size to resize the test sets to.
     */
    void resize_test(std::size_t new_size) {
      if (test_images.size() > new_size) {
        test_images.resize(new_size);
        test_labels.resize(new_size);
      }
    }
  };

  /*!
   * \brief Read a CIFAR 10 data file inside the given containers
   * \param images The container to fill with the labels
   * \param path The path to the label file
   * \param limit The maximum number of elements to read (0: no limit)
   */
  template <typename Images, typename Labels>
  void read_cifar10_file(Images& images, Labels& labels, const std::string& path, std::size_t limit) {

    static bool range_inited(false);
    static std::vector<int> cifar_single_entry_range(3073);

    if (limit && limit <= images.size()) {
      return;
    }

    if (!range_inited) {
      range_inited = true;
      std::iota(cifar_single_entry_range.begin(), cifar_single_entry_range.end(), 0);
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

    // Prepare the size for the new
    images.resize(images.size() + size);
    labels.resize(labels.size() + size);

    std::for_each(std::execution::par, cifar_single_entry_range.begin(), cifar_single_entry_range.end(), [&](int i) {
      labels[start + i] = buffer[i * 3073];

      TensorMap<Tensor<char, 3>> im_value(&buffer[i * 3073 + 1], 32, 32, 3);

      // tensors are column-major and the image we get is row-major.
      Tensor<float, 3> col_major = im_value.shuffle(array<Index, 3>{2, 1, 0}).cast<float>();
      images[i] = col_major;
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
  template <typename Images, typename Labels>
  void read_test(const std::string& folder, std::size_t limit, Images& images, Labels& labels) {
    read_cifar10_file(images, labels, folder + "/test_batch.bin", limit);
  }

  /*!
   * \brief Read all training data
   *
   * The dataset is assumed to be in a cifar-10 subfolder
   *
   * \param limit The maximum number of elements to read (0: no limit)
   * \param func The functor to create the image objects.
   */
  template <typename Images, typename Labels>
  void read_training(const std::string& folder, std::size_t limit, Images& images, Labels& labels) {
    read_cifar10_file(images, labels, folder + "/data_batch_1.bin", limit);
    read_cifar10_file(images, labels, folder + "/data_batch_2.bin", limit);
    read_cifar10_file(images, labels, folder + "/data_batch_3.bin", limit);
    read_cifar10_file(images, labels, folder + "/data_batch_4.bin", limit);
    read_cifar10_file(images, labels, folder + "/data_batch_5.bin", limit);
  }

  /*!
   * \brief Read all test data.
   *
   * The dataset is assumed to be in a cifar-10 subfolder
   *
   * \param limit The maximum number of elements to read (0: no limit)
   * \param func The functor to create the image objects.
   */
  template <typename Images, typename Labels>
  void read_test(std::size_t limit, Images& images, Labels& labels) {
    read_test(CIFAR_DATA_LOCATION, limit, images, labels);
  }

  /*!
   * \brief Read all training data
   *
   * The dataset is assumed to be in a cifar-10 subfolder
   *
   * \param limit The maximum number of elements to read (0: no limit)
   * \param func The functor to create the image objects.
   */
  template <typename Images, typename Labels>
  void read_training(std::size_t limit, Images& images, Labels& labels) {
    read_training(CIFAR_DATA_LOCATION, limit, images, labels);
  }
    
  template <template <typename...> class Container, typename Image, typename Label = uint8_t>
  CIFAR10_dataset<Container, Image, Label> read_dataset_3d(std::size_t training_limit = 0, std::size_t test_limit = 0) {
    CIFAR10_dataset<Container, Image, Label> dataset;

    read_training(training_limit, dataset.training_images, dataset.training_labels);
    read_test(test_limit, dataset.test_images, dataset.test_labels);

    return dataset;
  }

} //end of namespace cifar

#endif
