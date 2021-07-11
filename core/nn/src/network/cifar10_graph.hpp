#pragma once


#include "network_graph.hpp"


namespace EigenSinn {

  template <typename Scalar, typename Actual, typename Device_>
  class Cifar10 : public NetworkBase<Scalar, Device_, CrossEntropyLoss<Scalar, Actual, 2, Device_>, Adam> {

  public:
    Cifar10(int num_classes, float lr = 0.001) : NetworkBase() {
      auto x = add(new Input<float, Device_>);

      x = add(x, new Conv2d<Scalar, Device_, RowMajor>(DSizes<Index, 4>{6, 3, 5, 5}, Padding2D{ 0, 0 }, 1));
      x = add(x, new ReLU<Scalar, 4, Device_, RowMajor>);
      x = add(x, new MaxPooling<Scalar, 4, Device_, RowMajor>(DSizes<Index, 2>{2, 2}, 2));

      x = add(x, new Conv2d<Scalar, Device_, RowMajor>(DSizes<Index, 4>{16, 6, 5, 5}, Padding2D{ 0, 0 }, 1));
      x = add(x, new ReLU<Scalar, 4, Device_, RowMajor>);
      x = add(x, new MaxPooling<Scalar, 4, Device_, RowMajor>(DSizes<Index, 2>{2, 2}, 2));

      // get flat dimension by pushing a zero tensor through the network defined so far
      int flat_dim = 400;

      x = add(x, new Flatten<Scalar, Device_, RowMajor>);


      x = add(x, new Linear<Scalar, Device_, RowMajor>(flat_dim, 120));
      x = add(x, new ReLU<Scalar, 2, Device_, RowMajor>);

      x = add(x, new Linear<Scalar, Device_, RowMajor>(120, 84));
      x = add(x, new ReLU<Scalar, 2, Device_, RowMajor>);

      // cross-entropy loss includes the softmax non-linearity
      x = add(x, new Linear<Scalar, Device_, RowMajor>(84, num_classes));

      add_loss_and_compile(x, lr);

    }
  };

}