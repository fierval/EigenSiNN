#include "dataset.h"
#include "..\include\dataset.h"

// Code from: https://github.com/wichtounet/cifar-10


std::tuple<ImageContainer, LabelContainer> next_batch(ImageContainer &data, LabelContainer &labels, size_t batch_size, bool restart)
{

  static bool fst_batch(true);
  static ImageContainer::iterator it_cur;
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

  ImageContainer next_data_batch;
  LabelContainer next_label_batch;

  if (it_cur < data.end()) {
  
    ImageContainer::iterator data_end = it_cur + batch_size;
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

cifar::CIFAR10_dataset<std::vector, Tensor<float, 3>, uint8_t> read_cifar_dataset() {

  auto dataset = cifar::read_dataset_3d<std::vector, Tensor<float, 3>, uint8_t>();
  return dataset;
}
