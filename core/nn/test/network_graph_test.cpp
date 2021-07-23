#pragma once

#include <gtest/gtest.h>

#include <network/cifar10_graph.hpp>

using namespace EigenSinn;

class GraphTest : public ::testing::Test {
protected:
  void SetUp() override {

    rand_image.resize(cifar_dims);
    rand_image.setRandom();

    cifar10 = std::make_shared<Cifar10<float, std::uint8_t, ThreadPoolDevice>>(10, 0.001);

    // need to go through network at least once before saving
    // random labels
    std::vector<uint8_t> values(cifar_dims[0] * 10);
    std::transform(values.begin(), values.end(), values.begin(), [](uint8_t i) {return rand() % 10; });

    DeviceTensor<uint8_t, 2> rand_labels(values.data(), DSizes<Index, 2>{cifar_dims[0], 10});

    cifar10->step(rand_image.raw(), rand_labels.raw());
  }

  std::shared_ptr<Cifar10<float, std::uint8_t, ThreadPoolDevice>> cifar10;
  std::string input_node_name = "input.1";
  DSizes<Index, 4> cifar_dims{ 10, 3, 32, 32 };
  DeviceTensor<float, 4> rand_image;
};

//TEST_F(GraphTest, Create) {
//
//  cifar10->print_graph();
//  std::ofstream graphviz("c:\\temp\\gviz.dot", std::ios::binary);
//  cifar10->write_graphviz(graphviz);
//
//  cifar10->print_traversal();
//  std::cerr << "=======================================" << std::endl;
//  cifar10->print_traversal(false);
//  cifar10->save();
//}

TEST_F(GraphTest, LoadModel) {
  auto model = cifar10->save();
  model->dump("c:\\temp\\cifar10_graph.txt");
  model->flush("c:\\temp\\cifar10_graph.onnx");

  EigenModel m = EigenModel::FromFile("c:\\temp\\cifar10_graph.onnx");

  cifar10->load(m, false);

  cifar10->print_graph();
  std::ofstream graphviz("c:\\temp\\gviz.dot", std::ios::binary);
  cifar10->write_graphviz(graphviz);

  cifar10->print_traversal();
  std::cerr << "=======================================" << std::endl;
  cifar10->print_traversal(false);

  
}