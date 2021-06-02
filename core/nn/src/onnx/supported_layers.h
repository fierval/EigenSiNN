#pragma once

#include "onnx.proto3.pb.h"

#include <layers/batchnorm.hpp>
#include <layers/convolution.hpp>
#include <layers/convolutiontrans.hpp>
#include <layers/dropout.hpp>
#include <layers/flatten.hpp>
#include <layers/input.hpp>
#include <layers/linear.hpp>
#include <layers/maxpooling.hpp>
#include <layers/relu.hpp>
#include <layers/sigmoid.hpp>
#include <layers/softmax.hpp>
#include <layers/tanh.hpp>

#include <map>
#include <set>
using namespace Eigen;

namespace EigenSinn {

  template<typename Scalar, typename Device_>
  struct SupportedLayers {

    SupportedLayers(onnx::GraphProto& _graph) 
      : graph(_graph) {

    }

    enum Layers : int {
      BatchNormalization, Conv, ConvTranspose, Dropout, Flatten, Gemm, MaxPool, LeakyRelu, Relu, Sigmoid, Softmax, Tanh
    };

    std::vector<std::string> layers = {"BatchNormalization", "Conv", "ConvTranspose", "Dropout", "Flatten", "Gemm", "MaxPool", "LeakyRelu", "Relu", "Sigmoid",  "Softmax", "Tanh"};
        
    inline LayerBase<Scalar, Device_> * create_from_node(const onnx::NodeProto& node) {
      
      const std::string& op_type = node.op_type();

      // unsopported layer!
      auto it = std::find(layers.begin(), layers.end(), op_type);
      if (it == layers.end()) {
        return nullptr;
      }

      // get the actual layer
      Layers layer_type = static_cast<Layers>(it - layers.begin());
      switch (layer_type)
      {
      case: Layers::BatchNormalization :
        return BatchNormalization(node);

      case: Layers::Conv :
        return Conv(node);

      case: Layers::ConvTranspose :
        return ConvTranspose(node);

      case: Layers::Dropout :
        return Dropout(node);

      case: Layers::Flatten :
        return Flatten(node);

      case: Layers::Gemm :
        return Gemm(node);

      case: Layers::MaxPool :
        return MaxPool(node);

      case: Layers::LeakyRelu :
        return LeakyRelu(node);

      case: Layers::Relu :
        return Relu(node);

      case: Layers::Sigmoid :
        return Sigmoid(node);

      case: Layers::Softmax :
        return Softmax(node);

      case: Layers::Tanh :
        return Tanh(node);

      default:
        return nullptr;
      }

    }

  protected:
    inline LayerBase<Scalar, Device_>* BatchNormalization(const onnx::NodeProto& node) {


    }
    inline LayerBase<Scalar, Device_>* Conv(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* ConvTranspose(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* Dropout(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* Flatten(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* Gemm(const onnx::NodeProto& node) {

    }

    inline LayerBase<Scalar, Device_>* MaxPool(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* LeakyRelu(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* Relu(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* Sigmoid(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* Softmax(const onnx::NodeProto& node) {

    }
    inline LayerBase<Scalar, Device_>* Tanh(const onnx::NodeProto& node) {

    }

    onnx::GraphProto& graph;
  };
}