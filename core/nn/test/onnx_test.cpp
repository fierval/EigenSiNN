#include <iostream>
#include <gtest/gtest.h>
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

#include "network/cifar10_network.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class OnnxSave : public ::testing::Test {

  protected:

    void SetUp() override {

      net = std::make_shared<Cifar10<ThreadPoolDevice, RowMajor, false>>(cifar_dims, 10, 0.001);
      net->init();
    }

    std::shared_ptr<Cifar10<ThreadPoolDevice, RowMajor, false>> net;
    std::string input_node_name = "input.1";
    DSizes<Index, 4> cifar_dims{ 10, 3, 32, 32 };
  };

  TEST_F(OnnxSave, SaveCifarTraining) {

    EigenModel model;

    // need to go through network at least once before saving
    DeviceTensor<float, 4> rand_image(cifar_dims);
    rand_image.setRandom();

    // random labels
    std::vector<uint8_t> values(cifar_dims[0] * 10);
    std::transform(values.begin(), values.end(), values.begin(), [](uint8_t i) {return rand() % 10; });

    DeviceTensor<uint8_t, 2> rand_labels(values.data(), DSizes<Index, 2>{cifar_dims[0], 10});

    net->step(rand_image, rand_labels);
    net->save_to_onnx(model);

  }

  TEST_F(OnnxSave, SaveCifarTest) {

    EigenModel model;

    // need to go through network at least once before saving
    DeviceTensor<float, 4> rand_image(cifar_dims);
    rand_image.setRandom();

    // random labels
    std::vector<uint8_t> values(cifar_dims[0] * 10);
    std::transform(values.begin(), values.end(), values.begin(), [](uint8_t i) {return rand() % 10; });

    DeviceTensor<uint8_t, 2> rand_labels(values.data(), DSizes<Index, 2>{cifar_dims[0], 10});

    net->step(rand_image, rand_labels);
    net->save_to_onnx(model);

    model.flush("c:\\temp\\cifar10.onnx");
    model.dump("c:\\temp\\cifar10.txt");
  }

} // EigenSinnTest