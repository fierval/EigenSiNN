#include "dataset.h"

// Code from: https://github.com/wichtounet/mnist
mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> create_mnist_dataset()
{

  //MNIST_DATA_LOCATION set by MNIST in cmake
  mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, float, uint8_t>(MNIST_DATA_LOCATION);

  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

  return dataset;
}

std::tuple<DataContainer, LabelContainer> next_batch(DataContainer &data, LabelContainer &labels, size_t batch_size, bool restart)
{

  static bool fst_batch(true);
  static DataContainer::iterator it_cur;
  static LabelContainer::iterator it_label;

  // restart at the start of each new epoch
  if (restart) {
    fst_batch = true;
  }

  if (fst_batch) {
  
    fst_batch = false;
    it_cur = data.begin();
    it_label = labels.begin();
  }

  DataContainer next_data_batch;
  LabelContainer next_label_batch;

  if (it_cur < data.end()) {
  
    DataContainer::iterator data_end = it_cur + batch_size;
    LabelContainer::iterator label_end = it_label + batch_size;

    if (label_end > labels.end()) {
      label_end = labels.end();
      data_end = data.end();
    }

    next_data_batch.assign(it_cur, data_end);
    next_label_batch.assign(it_label, label_end);

    it_cur = data_end;
    it_label = label_end;
  }

  return std::make_tuple(next_data_batch, next_label_batch);
}
