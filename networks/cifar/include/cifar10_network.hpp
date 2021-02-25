#pragma once

#include <network/network.hpp>

using namespace EigenSinn;

template<typename Device_ = ThreadPoolDevice>
class Cifar10 : public NetBase<float, 4, CrossEntropyLoss<float, uint8_t, 2, Device_>, Device_> {

public:
  Cifar10(array<Index, 4> input_dims, int num_classes, float learning_rate) {

    // push back rvalues so we don't have to invoke the copy constructor
    add(new Input<float, 4, Device_>);

    add(new Conv2d<float, Device_>(array<Index, 4>{6, 3, 5, 5}, Padding2D{ 0, 0 }, 1), get_optimizer<4>(learning_rate));
    add(new ReLU<float, 4, Device_>);
    add(new MaxPooling<float, 4, Device_>(array<Index, 2>{2, 2}, 2));
    
    add(new Conv2d<float, Device_>(array<Index, 4>{16, 6, 5, 5}, Padding2D{ 0, 0 }, 1), get_optimizer<4>(learning_rate));
    add(new ReLU<float, 4, Device_>);
    add(new MaxPooling<float, 4, Device_>(array<Index, 2>{2, 2}, 2));

    // get flat dimension by pushing a zero tensor through the network defined so far
    int flat_dim = get_flat_dimension(array<Index, 4>{1, 3, 32, 32});

    add(new Flatten<float, Device_>);


    add(new Linear<float, Device_>(flat_dim, 120), get_optimizer<2>(learning_rate));
    add(new ReLU<float, 2, Device_>);

    add(new Linear<float, Device_>(120, 84), get_optimizer<2>(learning_rate));
    add(new ReLU<float, 2, Device_>);

    // cross-entropy loss includes the softmax non-linearity
    add(new Linear<float, Device_>(84, num_classes), get_optimizer<2>(learning_rate));
  }

protected:
  template <Index Rank>
  inline auto get_optimizer(float learning_rate) {
    return dynamic_cast<OptimizerBase<float, Device_>*>(new SGD<float, Rank, Device_>(learning_rate, 0, false));
  }
};
