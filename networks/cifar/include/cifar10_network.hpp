#pragma once

#include <network/network.hpp>

using namespace EigenSinn;

template<typename Device_ = ThreadPoolDevice>
class Cifar10 : public NetBase<float, 4, CrossEntropyLoss<float, uint8_t, 2, Device_>, Device_> {

public:
  Cifar10(array<Index, 4> input_dims, int num_classes, float learning_rate) {

    // push back rvalues so we don't have to invoke the copy constructor
    add(NetworkNode<float, Device_>(new Input<float, 4>));

    add(NetworkNode<float, Device_>(new Conv2d<float>(array<Index, 4>{6, 3, 5, 5}, Padding2D{ 0, 0 }, 1), get_optimizer<4>(learning_rate)));
    add(NetworkNode<float, Device_>(new ReLU<float, 4>));
    add(NetworkNode<float, Device_>(new MaxPooling<float, 4>(array<Index, 2>{2, 2}, 2)));
    
    add(NetworkNode<float, Device_>(new Conv2d<float>(array<Index, 4>{16, 6, 5, 5}, Padding2D{ 0, 0 }, 1), get_optimizer<4>(learning_rate)));
    add(NetworkNode<float, Device_>(new ReLU<float, 4>));
    add(NetworkNode<float, Device_>(new MaxPooling<float, 4>(array<Index, 2>{2, 2}, 2)));

    // get flat dimension by pushing a zero tensor through the network defined so far
    int flat_dim = get_flat_dimension(array<Index, 4>{1, 3, 32, 32});

    add(NetworkNode<float, Device_>(new Flatten<float>));


    add(NetworkNode<float, Device_>(new Linear<float>(flat_dim, 120), get_optimizer<2>(learning_rate)));
    add(NetworkNode<float, Device_>(new ReLU<float, 2>));

    add(NetworkNode<float, Device_>(new Linear<float>(120, 84), get_optimizer<2>(learning_rate)));
    add(NetworkNode<float, Device_>(new ReLU<float, 2>));

    // cross-entropy loss includes the softmax non-linearity
    add(NetworkNode<float, Device_>(new Linear<float>(84, num_classes), get_optimizer<2>(learning_rate)));
  }

protected:
  template <Index Rank>
  inline auto get_optimizer(float learning_rate) {
    return dynamic_cast<OptimizerBase<float>*>(new SGD<float, Rank>(learning_rate, 0, false));
  }
};
