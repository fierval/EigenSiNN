#pragma once

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
#include "model.h"

using namespace Eigen;

namespace EigenSinn {

  template<typename Scalar, typename Device_>
  struct OnnxLoader {

    OnnxLoader(const EigenModel& model) 
     : model(model) {

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
        return LoadBatchNormalization(node);

      case: Layers::Conv :
        return LoadConv(node);

      case: Layers::ConvTranspose :
        return LoadConvTranspose(node);

      case: Layers::Dropout :
        return LoadDropout(node);

      case: Layers::Flatten :
        return LoadFlatten(node);

      case: Layers::Gemm :
        return LoadGemm(node);

      case: Layers::MaxPool :
        return LoadMaxPool(node);

      case: Layers::LeakyRelu :
        return LoadLeakyRelu(node);

      case: Layers::Relu :
        return LoadRelu(node);

      case: Layers::Sigmoid :
        return LoadSigmoid(node);

      case: Layers::Softmax :
        return LoadSoftmax(node);

      case: Layers::Tanh :
        return LoadTanh(node);

      default:
        return nullptr;
      }

    }

  protected:
    inline LayerBase<Scalar, Device_>* LoadBatchNormalization(const onnx::NodeProto& node) {

      // epsilon
      auto eps = node.attribute(0).f();

      // momentum
      auto momentum = node.attribute(1).f();

      // find out the number of features by looking into the initializers
      // and figuring out what the weight dimension is
      const std::string& weight_name = node.input().Get(1);
      Index num_features = model.get_input_dimensions(weight_name)[0];
      
      // create layer
      auto * out =  new BatchNormalizationLayer<Scalar, Device_, RowMajor>(num_features, eps, momentum);
      out->load_onnx_data(model, get_node_inputs(node));
      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadConv(const onnx::NodeProto& node) {

      DSizes<Index, 4> kernel_dims;
      Padding2D padding;
      int stride;
      int dilation;

      std::tie(kernel_dims, padding, stride, dilation) = get_conv_node_attributes(node);

      auto* out = new Conv2d<Scalar, Device_, RowMajor>(kernel_dims, padding, stride, dilation);
      out->load_onnx_data(model, get_node_inputs(node));
      return out;

    }

    inline LayerBase<Scalar, Device_>* LoadConvTranspose(const onnx::NodeProto& node) {

      DSizes<Index, 4> kernel_dims;
      Padding2D padding;
      int stride;
      int dilation;

      std::tie(kernel_dims, padding, stride, dilation) = get_conv_node_attributes(node);

      auto * out = new TransConv2d<Scalar, Device_, RowMajor>(kernel_dims, padding, stride, dilation);
      out->load_onnx_data(model, get_node_inputs(node));
      return out;

    }

    inline LayerBase<Scalar, Device_>* LoadDropout(const onnx::NodeProto& node) {

      // dropout probability is saved as input
      auto inputs = get_node_inputs(node);
      std::vector<Scalar*> prob_data;
      std::vector<std::vector<Index>> prob_data_dims;

      std::tie(prob_data, prob_data_dims) = model.get_input_data_and_dimensions(inputs);

      // TODO: How do we deal with Rank?
      auto* out = new EigenSinn::Dropout<Scalar, 4, Device_, RowMajor>(*prob_data[0]);
      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadFlatten(const onnx::NodeProto& node) {
      // flatten is always agains dim 1
      return new EigenSinn::Flatten<Scalar, Device_, RowMajor>();
    }

    inline LayerBase<Scalar, Device_>* LoadGemm(const onnx::NodeProto& node) {

      auto inputs = get_node_inputs(node);

      std::vector<Scalar*> data;
      std::vector<std::vector<Index>> dims;

      std::tie(data, dims) = model.get_input_data_and_dimensions(inputs);

      // dimesions of the weight tensor
      int in_dim = data[1][0], out_dim[1][1];
      auto* out = new Linear<Scalar, Device_, RowMajor>(in_dim, out_dim);
      out->load_onnx_data(model, inputs);

      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadMaxPool(const onnx::NodeProto& node) {
      // TODO: How do we deal with Rank?
      auto* out = new MaxPooling<Scalar, 4, Device_, RowMajor>();
      return out;

    }

    inline LayerBase<Scalar, Device_>* LoadLeakyRelu(const onnx::NodeProto& node) {
      
      auto attrs = get_node_attributes(node);
      // TODO: How do we deal with Rank?
      return new EigenSinn::LeakyReLU<Scalar, 4, Device_, RowMajor>(attrs["alpha"]->f());
    }

    inline LayerBase<Scalar, Device_>* LoadRelu(const onnx::NodeProto& node) {

      // TODO: How do we deal with Rank?
      return new EigenSinn::ReLU<Scalar, 4, Device_, RowMajor>();
    }

    inline LayerBase<Scalar, Device_>* LoadSigmoid(const onnx::NodeProto& node) {
      return new EigenSinn::Sigmoid<Scalar, 4, Device_, RowMajor>();

    }
    inline LayerBase<Scalar, Device_>* LoadSoftmax(const onnx::NodeProto& node) {
      return new EigenSinn::Softmax<Scalar, 4, Device_, RowMajor>();

    }

    inline LayerBase<Scalar, Device_>* LoadTanh(const onnx::NodeProto& node) {
      return new EigenSinn::Tanh<Scalar, 4, Device_, RowMajor>();
    }

    std::vector<std::string> get_node_inputs(const onnx::NodeProto& node) {

      std::vector<std::string> out(node.input_size());

      std::transform(node.input().begin(), node.input().end(), out.begin(), [](std::string& s) {return s; });
    }

    std::map<std::string, onnx::AttributeProto*> get_node_attributes(const onnx::NodeProto& node) {

      std::map<std::string, onnx::AttributeProto*> attr;

      std::transform(node.attribute().begin(), node.attribute().end(), std::back_inserter(attr), [](onnx::AttributeProto& a) {return std::pair<std::string, onnx::AttributeProto*>(a.name(), &a); });
      return out;
    }

    std::tuple<DSizes<Index, 4>, Padding2D, int, int> get_conv_node_attributes(const onnx::NodeProto& node) {

      auto attrs = get_node_attributes(node);

      DSizes<Index, 4> kernel_dims = vec2dims<4>(get_attr_dims(*attrs["kernel_shape"]));
      Padding2D padding{ attrs["padding"]->ints().Get(2), attrs["padding"]->ints().Get(3) };
      int stride = attrs["strides"]->ints().Get(0);
      int dilation = attrs["dilations"]->ints().Get(0);

      return std::make_tuple(kernel_dims, padding, stride, dilation);

    }

    const std::vector<Index> get_attr_dims(onnx::AttributeProto& attr) {
      std::vector<Index> dims(attr.ints().size());
      std::transform(attr.ints().begin(), attr.ints().end(), dims.begin(), [](Index i) {return i; });
      return dims;
    }

    const EigenModel& model;
  };
}