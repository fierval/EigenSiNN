#include <iostream>
#include <gtest/gtest.h>
#include <layers/relu.hpp>
#include <layers/input.hpp>
#include <layers/convolution.hpp>
#include "include/commondata4d.hpp"
#include "include/testutils.hpp"
#include "ops/comparisons.hpp"

#include <onnx/onnx.proto3.pb.h>
#include <onnx/common.h>

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

    }

    std::shared_ptr<Conv2d<float, ThreadPoolDevice, RowMajor>> conv;
    Input<float, 4, ThreadPoolDevice, RowMajor> input;
    CommonData4d<ThreadPoolDevice, RowMajor> cd;
  };

  TEST_F(OnnxSave, SaveModel) {

    // Model
    EigenModel model;
    model.add_input("input", input.get_dims(), data_type_from_scalar<float>());
    model.add_output("output", conv->out_dims(), data_type_from_scalar<float>());

    // Node
    std::unique_ptr<onnx::NodeProto> graph = std::make_unique<onnx::NodeProto>();
  }
} // EigenSinnTest