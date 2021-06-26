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
#include "op_defs.h"

#define make_func(a) std::bind((&OnnxLoader<Scalar, Device_>::a), this, std::placeholders::_1)

using namespace Eigen;

namespace EigenSinn {

  template<typename Scalar, typename Device_>
  struct OnnxLoader {

    OnnxLoader(EigenModel& model) 
     : model(model) {

      layer_loaders.insert(std::make_pair(batch_norm_op, make_func(LoadBatchNormalization)));
      layer_loaders.insert(std::make_pair(conv_op, make_func(LoadConv)));
      layer_loaders.insert(std::make_pair(conv_transpose_op, make_func(LoadConvTranspose)));
      layer_loaders.insert(std::make_pair(dropout_op, make_func(LoadDropout)));
      layer_loaders.insert(std::make_pair(flatten_op, make_func(LoadFlatten)));
      layer_loaders.insert(std::make_pair(gemm_op, make_func(LoadGemm)));
      layer_loaders.insert(std::make_pair(maxpool_op, make_func(LoadMaxPool)));
      layer_loaders.insert(std::make_pair(leakyrelu_op, make_func(LoadLeakyRelu)));
      layer_loaders.insert(std::make_pair(relu_op, make_func(LoadRelu)));
      layer_loaders.insert(std::make_pair(sigmoid_op, make_func(LoadSigmoid)));
      layer_loaders.insert(std::make_pair(softmax_op, make_func(LoadSoftmax)));
      layer_loaders.insert(std::make_pair(tanh_op, make_func(LoadTanh)));
    }

    inline LayerBase<Scalar, Device_> * create_from_node(const onnx::NodeProto& node) {
      
      const std::string& op_type = node.op_type();

      // unsopported layer!
      if (layer_loaders.find(op_type.c_str()) == layer_loaders.end()) {
        return nullptr;
      }

      // get the actual layer
      return layer_loaders[op_type.c_str()](node);
    }

  protected:
    inline LayerBase<Scalar, Device_>* LoadBatchNormalization(const onnx::NodeProto& node) {

      // epsilon
      auto eps = node.attribute(0).f();

      // momentum
      auto momentum = node.attribute(1).f();

      // find out the number of features by looking into the initializers
      // and figuring out what the weight dimension is
      Index num_features = model.get_input_dimensions(node.input().Get(1))[0];
      
      auto rank_attr = get_node_attributes(node)["rank"].i();

      // create layer
      LayerBase<Scalar, Device_>* out;
      
      switch (rank_attr)
      {
      case 2:
        out = new BatchNormalizationLayer<Scalar, 2, Device_, RowMajor>(num_features, eps, momentum);
        break;
      case 4:
        out = new BatchNormalizationLayer<Scalar, 4, Device_, RowMajor>(num_features, eps, momentum);
        break;

      default:
        throw std::logic_error("Unsupported rank");
        break;
      }
      
      out->load_onnx_data(model, get_node_inputs(node));
      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadConv(const onnx::NodeProto& node) {

      DSizes<Index, 2> kernel_dims;
      Padding2D padding;
      int stride;
      int dilation;

      std::tie(kernel_dims, padding, stride, dilation) = get_conv_node_attributes(node);

      auto inputs = get_node_inputs(node);

      std::vector<std::vector<Index>> dimensions;
      std::vector<Scalar*> values;

      // kernel dimensions are second
      DSizes<Index, 4> real_kernel_dims;
      std::tie(values, dimensions) = model.get_input_data_and_dimensions<Scalar>(inputs);

      // consistent between attributes and actual dimensions
      assert(kernel_dims[0] == real_kernel_dims[2] && kernel_dims[1] == real_kernel_dims[3]);

      auto* out = new Conv2d<Scalar, Device_, RowMajor>(real_kernel_dims, padding, stride, dilation);
      out->load_onnx_data(model, inputs);
      return out;

    }

    inline LayerBase<Scalar, Device_>* LoadConvTranspose(const onnx::NodeProto& node) {

      DSizes<Index, 2> kernel_dims;
      Padding2D padding;
      int stride;
      int dilation;

      std::tie(kernel_dims, padding, stride, dilation) = get_conv_node_attributes(node);

      auto inputs = get_node_inputs(node);

      std::vector<std::vector<Index>> dimensions;
      std::vector<Scalar*> values;

      // kernel dimensions are second
      DSizes<Index, 4> real_kernel_dims;
      std::tie(values, dimensions) = model.get_input_data_and_dimensions<Scalar>(inputs);

      // consistent between attributes and actual dimensions
      assert(kernel_dims[0] == real_kernel_dims[2] && kernel_dims[1] == real_kernel_dims[3]);

      auto * out = new TransConv2d<Scalar, Device_, RowMajor>(real_kernel_dims, padding, stride, dilation);
      out->load_onnx_data(model, inputs);
      return out;

    }

    inline LayerBase<Scalar, Device_>* LoadDropout(const onnx::NodeProto& node) {

      // dropout probability is saved as input
      auto inputs = get_node_inputs(node);
      std::vector<Scalar*> prob_data;
      std::vector<std::vector<Index>> prob_data_dims;

      // TODO: BUG!!! There may not be an initializer. Change to get_input_dimensions()
      std::tie(prob_data, prob_data_dims) = model.get_input_data_and_dimensions<Scalar>(inputs);

      auto rank_attr = get_node_attributes(node)["rank"].i();

      // create layer
      LayerBase<Scalar, Device_>* out;

      switch (rank_attr)
      {
      case 2:
        out = new Dropout<Scalar, 2, Device_, RowMajor>(*prob_data[0]);
        break;
      case 4:
        out = new Dropout<Scalar, 4, Device_, RowMajor>(*prob_data[0]);
        break;

      default:
        throw std::logic_error("Unsupported rank");
        break;
      }

      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadFlatten(const onnx::NodeProto& node) {
      // flatten is always agains dim 1
      return new EigenSinn::Flatten<Scalar, Device_, RowMajor>();
    }

    inline LayerBase<Scalar, Device_>* LoadGemm(const onnx::NodeProto& node) {

      auto inputs = get_node_inputs(node);

      std::vector<std::vector<Index>> dimensions;
      std::vector<Scalar*> values;

      std::tie(values, dimensions) = model.get_input_data_and_dimensions<Scalar>(inputs);

      // dimesions of the weight tensor
      int in_dim = dimensions[1][0], out_dim = dimensions[1][1];
      auto* out = new Linear<Scalar, Device_, RowMajor>(in_dim, out_dim);
      out->load_onnx_data(model, inputs);

      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadMaxPool(const onnx::NodeProto& node) {

      DSizes<Index, 2> kernel_dims;
      Padding2D padding;
      int stride;
      
      std::tie(kernel_dims, padding, stride) = get_conv_node_common_attributes(node);

      auto rank_attr = get_node_attributes(node)["rank"].i();

      // create layer
      LayerBase<Scalar, Device_>* out;

      switch (rank_attr)
      {
      case 2:
        out = new MaxPooling<Scalar, 4, Device_, RowMajor>(kernel_dims, stride, padding);
        break;
      case 4:
        out = new MaxPooling<Scalar, 4, Device_, RowMajor>(kernel_dims, stride, padding);
        break;

      default:
        throw std::logic_error("Unsupported rank");
        break;
      }

      return out;

    }

    inline LayerBase<Scalar, Device_>* LoadLeakyRelu(const onnx::NodeProto& node) {
      
      auto attrs = get_node_attributes(node);
      auto rank_attr = attrs["rank"].i();

      // create layer
      LayerBase<Scalar, Device_>* out;

      switch (rank_attr)
      {
      case 2:
        out = new LeakyReLU<Scalar, 2, Device_, RowMajor>(attrs["alpha"].f());
        break;
      case 4:
        out = new LeakyReLU<Scalar, 4, Device_, RowMajor>(attrs["alpha"].f());
        break;

      default:
        throw std::logic_error("Unsupported rank");
        break;
      }

      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadRelu(const onnx::NodeProto& node) {

      auto rank_attr = get_node_attributes(node)["rank"].i();

      // create layer
      LayerBase<Scalar, Device_>* out;

      switch (rank_attr)
      {
      case 2:
        out = new ReLU<Scalar, 2, Device_, RowMajor>();
        break;
      case 4:
        out = new ReLU<Scalar, 4, Device_, RowMajor>();
        break;

      default:
        throw std::logic_error("Unsupported rank");
        break;
      }

      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadSigmoid(const onnx::NodeProto& node) {
      auto rank_attr = get_node_attributes(node)["rank"].i();

      // create layer
      LayerBase<Scalar, Device_>* out;

      switch (rank_attr)
      {
      case 2:
        out = new Sigmoid<Scalar, 2, Device_, RowMajor>();
        break;
      case 4:
        out = new Sigmoid<Scalar, 4, Device_, RowMajor>();
        break;

      default:
        throw std::logic_error("Unsupported rank");
        break;
      }
      return out;

    }

    inline LayerBase<Scalar, Device_>* LoadSoftmax(const onnx::NodeProto& node) {
      auto rank_attr = get_node_attributes(node)["rank"].i();

      // create layer
      LayerBase<Scalar, Device_>* out;

      switch (rank_attr)
      {
      case 2:
        out = new Softmax<Scalar, 2, Device_, RowMajor>();
          break;
      case 4:
        out = new Softmax<Scalar, 4, Device_, RowMajor>();
          break;

      default:
        throw std::logic_error("Unsupported rank");
        break;
      }
      return out;
    }

    inline LayerBase<Scalar, Device_>* LoadTanh(const onnx::NodeProto& node) {
      auto rank_attr = get_node_attributes(node)["rank"].i();

      // create layer
      LayerBase<Scalar, Device_>* out;

      switch (rank_attr)
      {
      case 2:
        out = new Tanh<Scalar, 2, Device_, RowMajor>();
          break;
      case 4:
        out = new Tanh<Scalar, 4, Device_, RowMajor>();
          break;

      default:
        throw std::logic_error("Unsupported rank");
        break;
      }
      return out;
    }

    std::vector<std::string> get_node_inputs(const onnx::NodeProto& node) {

      std::vector<std::string> out(node.input_size());

      std::transform(node.input().begin(), node.input().end(), out.begin(), [](const std::string& s) {return s; });
      return out;
    }

    std::map<const std::string, onnx::AttributeProto> get_node_attributes(const onnx::NodeProto& node) {

      std::map<const std::string, onnx::AttributeProto> attr;

      std::transform(node.attribute().begin(), node.attribute().end(), std::inserter(attr, attr.begin()), [](const onnx::AttributeProto& a) {return std::pair<const std::string, onnx::AttributeProto>(a.name(), a); });
      return attr;
    }

    // kernel_shape is only good for debugging as we don't get all the dimensions
    std::tuple<DSizes<Index, 2>, Padding2D, int, int> get_conv_node_attributes(const onnx::NodeProto& node) {

      auto attrs = get_node_attributes(node);

      DSizes<Index, 2> kernel_dims;

      Padding2D padding;
      int stride;

      std::tie(kernel_dims, padding, stride) = get_conv_node_common_attributes(node);
      int dilation = attrs["dilations"].ints().Get(0);

      return std::make_tuple(kernel_dims, padding, stride, dilation);

    }

    // Attributes in common for maxpool and conv
    std::tuple<DSizes<Index, 2>, Padding2D, int> get_conv_node_common_attributes(const onnx::NodeProto& node) {

      auto attrs = get_node_attributes(node);

      DSizes<Index, 2> kernel_dims = vec2dims<2>(get_attr_dims(attrs["kernel_shape"]));
      Padding2D padding{ (long)attrs["pads"].ints().Get(2), (long)attrs["pads"].ints().Get(3) };
      int stride = attrs["strides"].ints().Get(0);

      return std::make_tuple(kernel_dims, padding, stride);

    }

    const std::vector<Index> get_attr_dims(const onnx::AttributeProto& attr) {
      std::vector<Index> dims(attr.ints().size());
      std::transform(attr.ints().begin(), attr.ints().end(), dims.begin(), [](Index i) {return i; });
      return dims;
    }

    EigenModel& model;

    std::map<const char*, std::function<LayerBase<Scalar, Device_>* (const onnx::NodeProto&)>> layer_loaders;

  };
}