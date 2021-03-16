#pragma once

#include "dataset.h"

#include <losses/crossentropyloss.hpp>
#include <device/device_tensor.hpp>
#include <chrono>
#include <vector>
#include <string>
#include "cifar10_network.hpp"

using namespace EigenSinn;

void move_to_host(Cifar10<GpuDevice>& dev_net, Cifar10<DefaultDevice>& host_net) {

  // conv1, conv2, linear1, linear2, linear3
  std::vector<int> layers{ 1, 4, 8, 10, 11 };
  auto it = layers.begin();

  for_each(it, it + 2, [&](int i) {

    DeviceTensor<float, 4, GpuDevice, ColMajor> conv_t_w(dev_net.network[i].layer->get_weights());
    DeviceTensor<float, 4, DefaultDevice, ColMajor> h_conv_t_w(conv_t_w.to_host());
    host_net.network[i].layer->set_weights(h_conv_t_w.raw());
    });

  it = layers.begin() + 2;
  for_each(it, layers.end(), [&](int i) {

    DeviceTensor<float, 2, GpuDevice, ColMajor> linear_t_w(dev_net.network[i].layer->get_weights());
    DeviceTensor<float, 2, DefaultDevice, ColMajor> h_linear_t_w(linear_t_w.to_host());
    host_net.network[i].layer->set_weights(h_linear_t_w.raw());
    });


  for_each(layers.begin(), layers.end(), [&](int i) {

    DeviceTensor<float, 1, GpuDevice, ColMajor> layer_t_w(dev_net.network[i].layer->get_bias());
    DeviceTensor<float, 1, DefaultDevice, ColMajor> h_layer_t_w(layer_t_w.to_host());
    host_net.network[i].layer->set_bias(h_layer_t_w.raw());
    });
}
