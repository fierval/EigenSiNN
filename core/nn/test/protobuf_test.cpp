#include <iostream>
#include <gtest/gtest.h>
#include <layers/relu.hpp>
#include <layers/input.hpp>
#include <layers/convolution.hpp>
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

#include "include/cifar10_network.hpp"

using namespace EigenSinn;

namespace EigenSinnTest {

  class OnnxSave : public ::testing::Test {

  protected:

    void SetUp() override {

      cd.init();

      // Create Model
      input.set_input(cd.convInput);

      conv = std::make_shared<Conv2d<float, ThreadPoolDevice, RowMajor>>(cd.kernelDims);
      conv->init();
      conv->forward(input);

      net = std::make_shared<Cifar10<ThreadPoolDevice, RowMajor, false>>(cd.dims, 10, 0.001);
      net->init();
    }

    std::shared_ptr<Conv2d<float, ThreadPoolDevice, RowMajor>> conv;
    ReLU<float, 4, ThreadPoolDevice, RowMajor> relu;
    Input<float, 4, ThreadPoolDevice, RowMajor> input;
    CommonData4d<ThreadPoolDevice, RowMajor> cd;

    std::shared_ptr<Cifar10<ThreadPoolDevice, RowMajor, false>> net;
    std::string input_node_name = "input.1";
  };

  TEST_F(OnnxSave, SaveModel) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // Model
    EigenModel model;

    Conv2d<float, ThreadPoolDevice, RowMajor>& conv = *(this->conv);

    model.add_input("input", input.get_dims(), onnx_data_type_from_scalar<float>());

    auto output_name = conv.add_onnx_node(model, "input");

    model.add_output(output_name, conv.onnx_out_dims(), onnx_data_type_from_scalar<float>());

    model.flush("c:\\temp\\test.onnx");
    model.dump("c:\\temp\\test.txt");
  }

  TEST_F(OnnxSave, SaveCifar) {

    EigenModel model;

    net->save_to_onnx();

    model.flush("c:\\temp\\cifar10.onnx");
    model.dump("c:\\temp\\cifar10.txt");
  }
} // EigenSinnTest