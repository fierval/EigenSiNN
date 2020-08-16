#pragma once

#include <mnist/mnist_reader.hpp>
using DataContainer = std::vector<std::vector<uint8_t>>;
using LabelContainer = std::vector<uint8_t>;

mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> create_mnist_dataset();
std::tuple<DataContainer, LabelContainer> next_batch(DataContainer& data, LabelContainer& labels, size_t batch_size);