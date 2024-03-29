#pragma once

#include "layer_base.hpp"
#include <ops/batchnorm.hpp>
#include <device/device_tensor.hpp>

#include <onnx/op_defs.h>

using namespace  Eigen;
using std::unique_ptr;
using std::make_unique;

namespace EigenSinn {

  // NCHW format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class BatchNormalizationLayer : public LayerBase<Scalar, Device_> {
  public:

    BatchNormalizationLayer(Index num_features, float _eps = 1e-5, float _momentum = 0.9, bool _is_training = true)
      : LayerBase<Scalar, Device_>(OnnxOpNames::batch_norm_op)
      , momentum(_momentum)
      , eps(_eps)
      , is_training(_is_training)
      , beta(num_features)
      , dgamma(num_features)
      , dbeta(num_features)
      , gamma(num_features)
      , running_variance(num_features)
      , running_mean(num_features)
      , mu(num_features)
      , var(num_features) {
    }

    void init() override {
      beta.setZero();
      gamma.setConstant(1.);
      running_variance.setZero();
      running_mean.setZero();
    }

    void init(TensorSingleDim<Scalar>& _beta, TensorSingleDim<Scalar>& _gamma) {
      init();

      beta.set_from_host(_beta);
      gamma.set_from_host(_gamma);
    }

    void forward(PtrTensorAdaptor<Scalar, Device_>& prev_layer_base) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> prev_layer(prev_layer_base);

      if (!xhat) {
        DSizes<Index, Rank> dims = prev_layer.dimensions();
        layer_gradient.resize(dims);
        layer_output.resize(dims);
        xhat.resize(dims);
      }


      std::tie(layer_output, xhat, running_mean, running_variance, mu, var) =
        batch_norm(prev_layer, gamma, beta, eps, momentum, running_mean, running_variance, is_training);
    }

    // see https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    // for derivations
    void backward(PtrTensorAdaptor<Scalar, Device_>& prev_layer_any, PtrTensorAdaptor<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> prev_layer(prev_layer_any);
      DeviceTensor<Scalar, Rank, Device_, Layout> dout(next_layer_grad_any);

      DSizes<Index, Rank - 1> reduction_dims;
      DSizes<Index, Rank> broadcast_dims;

      float total_channel = 1.;
      for (int i = 0; i < Rank; i++) {
        if (i == (int)ImageDims::channel) {
          continue;
        }
        total_channel *= prev_layer->dimension(i);
      }

      std::tie(reduction_dims, broadcast_dims) = get_broadcast_and_reduction_dims<Scalar, Rank>(*dout);

      //broadcast values
      DeviceTensor<Scalar, Rank, Device_, Layout> broadcast_mean = broadcast_as_last_dim(mu, broadcast_dims);
      DeviceTensor<Scalar, Rank, Device_, Layout> broadcast_var = broadcast_as_last_dim(var, broadcast_dims);
      DeviceTensor<Scalar, Rank, Device_, Layout> xmu(prev_layer.dimensions());

      xmu.view() = *prev_layer - *broadcast_mean;

      // Step 9
      // dbeta = sum(dout, reduced by all dims except channel)
      Device_& device(dbeta.device());

      dbeta.view() = dout->sum(reduction_dims);

      // Step 8
      // dgamma = sum (dout * y, reduced by all dims except channel)
      DeviceTensor<Scalar, Rank, Device_, Layout> gamma_broad(broadcast_as_last_dim(gamma, broadcast_dims));
      DeviceTensor<Scalar, Rank, Device_, Layout> dxhat(dout.dimensions());
      dxhat.view() = *dout * *gamma_broad;
      dgamma.view() = (*dout * *xhat).sum(reduction_dims);

      // Step 7
      // d_inv_std
      //TensorSingleDim d_inv_std = (dxhat * xmu).sum(reduction_dims);
      DeviceTensor<Scalar, Rank, Device_, Layout> dxmu1(dxhat.dimensions());
      dxmu1.view() = *dxhat * (1. / (*broadcast_var + eps).sqrt());

      // Step 6
      // TensorSingleDim d_std = -d_inv_std / (running_var + eps);

      // Step 5
      DeviceTensor<Scalar, 1, Device_> d_var(var.dimensions());
      d_var.view() = -0.5 * (*dxhat * *xmu).sum(reduction_dims) / (*var + eps).pow(3. / 2.);

      // Step 4
      DeviceTensor<Scalar, Rank, Device_, Layout> d_var_broadcast = broadcast_as_last_dim(d_var, broadcast_dims);
      DeviceTensor<Scalar, Rank, Device_, Layout> d_sq(dout.dimensions());

      d_sq.view() = 1. / total_channel * *d_var_broadcast;

      // Step 3
      DeviceTensor<Scalar, Rank, Device_, Layout> dxmu2(d_sq.dimensions());
      dxmu2.view() = 2 * *xmu * *d_sq;

      // step 2
      DeviceTensor<Scalar, Rank, Device_, Layout> dx1(dxmu1.dimensions());
      dx1.view() = *dxmu1 + *dxmu2;

      DeviceTensor<Scalar, 1, Device_, Layout> dmu(dout->dimension(1));
      dmu.view() = -dx1->sum(reduction_dims);

      // step 1
      DeviceTensor<Scalar, Rank, Device_, Layout> dx2(dout.dimensions());
      DeviceTensor<Scalar, Rank, Device_, Layout> dmu_broadcast = broadcast_as_last_dim(dmu, broadcast_dims);

      dx2.view() = 1. / total_channel * *dmu_broadcast;

      // step 0
      layer_gradient.view() = *dx1 + *dx2;
    }

    PtrTensorAdaptor<Scalar, Device_> get_output() {

      return layer_output.raw();
    }

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative() {
      return layer_gradient.raw();
    }

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_weights_derivative() override {
      return dgamma.raw();
    }

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_bias_derivative() override {
      return dbeta.raw();
    }

    inline void SetTraining(bool training) {
      is_training = training;
    }

    inline bool IsTraining() {
      return is_training;
    }

    PtrTensorAdaptor<Scalar, Device_> get_weights() override {
      return gamma.raw();
    }

    PtrTensorAdaptor<Scalar, Device_> get_bias() override {
      return beta.raw();
    }

    void set_weights(PtrTensorAdaptor<Scalar, Device_>& v) override {
      gamma = DeviceTensor<Scalar, 1, Device_, Layout>(v);
    }

    void set_bias(PtrTensorAdaptor<Scalar, Device_>& v) override {
      beta = DeviceTensor<Scalar, 1, Device_, Layout>(v);
    }

    inline const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override {

      // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#BatchNormalization
      static constexpr char s_batch[] = "batch";

      // 1. ADd ONNX node with inputs & outputs
      std::vector<const char*> suffixes{"weight", "bias", "input_mean", "input_var" };
      std::vector<std::string> names = model.get_cool_display_tensor_value_names(s_batch, suffixes);
      
      names.insert(names.begin(), input_name);

      onnx::NodeProto* node = model.add_graph_node(get_layer_name(), get_op_name(), names);
      const std::string out_name = node->output().Get(0);

      // 2. Attributes
      model.add_attr(node, "epsilon", eps);
      model.add_attr(node, "momentum", momentum);

      // Rank attribute (not part of ONNX)
      model.add_attr(node, "rank", Rank);

      // 3. Initializers
      gamma.save_onnx_initializer(model, names[0]);
      beta.save_onnx_initializer(model, names[1]);
      running_mean.save_onnx_initializer(model, names[2]);
      running_variance.save_onnx_initializer(model, names[3]);

      return out_name;
    }

    const std::vector<Index> onnx_out_dims() override {
      return layer_output.vec_dims();
    }

    // in the order they appear in the ONNX file
    inline void load_onnx_data(EigenModel& model, std::vector<std::string>& inputs) override {

      std::vector<std::vector<Index>> dimensions;
      std::vector<onnx::TensorProto> values;

      std::tie(values, dimensions) = model.get_input_data_and_dimensions<Scalar>(inputs);

      // even though all dimensions are the same, we iterate over them for completeness
      gamma.set_from_host(model.get_input_data<Scalar>(values[0]), vec2dims<1>(dimensions[0]));
      beta.set_from_host(model.get_input_data<Scalar>(values[1]), vec2dims<1>(dimensions[1]));
      running_mean.set_from_host(model.get_input_data<Scalar>(values[2]), vec2dims<1>(dimensions[2]));
      running_variance.set_from_host(model.get_input_data<Scalar>(values[3]), vec2dims<1>(dimensions[3]));

      // inputs are stored weights only, exclude the actual input tensor
      gamma.set_node_input_name(inputs[1]);
      beta.set_node_input_name(inputs[2]);
      running_mean.set_node_input_name(inputs[3]);
      running_variance.set_node_input_name(inputs[4]);

    }

  private:

    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_gradient, xhat;

    DeviceTensor<Scalar, 1, Device_, Layout> gamma, beta, running_mean, running_variance, mu, var;
    DeviceTensor<Scalar, 1, Device_, Layout> dbeta, dgamma;
    Scalar momentum, eps;
    bool is_training;

  };

}