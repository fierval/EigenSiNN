#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"
#include "ops/initializations.hpp"
#include "helpers/conv_params_bag.hpp"
#include <onnx/op_defs.h>

#ifdef __CUDACC__
#include "cudnn/cudnn_workspace.hpp"
#endif

namespace EigenSinn {

  // REVIEW: Not implementing bias for now
  // Batch normalization layers can take care of bias
  template <typename Scalar, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class TransConv2d : public LayerBase<Scalar, Device_> {

  public:

    TransConv2d(const array<Index, 4>& kernelDims, const Padding2D& _padding = { 0, 0 }, const Index _stride = 1, const Index _dilation = 1)
      : LayerBase<Scalar, Device_>(conv_transpose_op)
      , kernel(kernelDims)
      , padding(_padding)
      , stride(_stride)
      , dilation(_dilation)
      , bias(kernelDims[1])
      , bias_broadcast({ 0, 0, 0, 0 })
      , loss_by_bias_derivative(kernelDims[1]) {
    
    }


    // TODO: this needs to be implemented for real
    // Also strides and padding should be added
    void init() override {

      // Wrapping the pointer and moving it to the tensor keeping in mind the GPU device
      kernel = generate_xavier<Scalar, 4, Layout, Device_>(kernel.dimensions());
      bias.setZero();
    }

    void init(Tensor<Scalar, 4, Layout>& _weights) {
      init();

      kernel.set_from_host(_weights);
    }

    void forward(PtrTensorAdapter<Scalar, Device_>& prev_layer_any) override {

      DeviceTensor<Scalar, 4, Device_, Layout> prev_layer(prev_layer_any);

      if (!params || params->check(prev_layer.dimensions())) {
        params = std::make_shared<ConvolutionParams<4>>(prev_layer.dimensions(), kernel.dimensions(), padding, stride, dilation, true, is_cudnn);

        dX.resize(params->get_input_dims());
        dW.resize(kernel.dimensions());
        layer_output.resize(params->get_out_dims());

#ifdef __CUDACC__
        if (is_cudnn) {
          cudnn_workspace.reset(new CudnnWorkspace(*params));
        }
#endif

      }

      DSizes<Index, 4> out_dims = params->orig_dims();
      //add bias to each channel
      bias_broadcast = { out_dims[0], 1, out_dims[2], out_dims[3] };

#ifdef __CUDACC__

      if (is_cudnn) {
        // data forward
        checkCudnnErrors(cudnnConvolutionBackwardData(CudnnWorkspace::cudnn(), &(CudnnWorkspace::one), cudnn_workspace->filter_desc, kernel->data(),
          cudnn_workspace->output_desc, prev_layer->data(), cudnn_workspace->conv_desc, cudnn_workspace->conv_bwd_data_algo,
          CudnnWorkspace::workspace(), CudnnWorkspace::workspace_size, &(CudnnWorkspace::zero), cudnn_workspace->input_desc, layer_output->data()));

      }
      else {
#endif // __CUDACC__

        DeviceTensor<Scalar, 2, Device_, Layout> inp_reshaped = unfold_conv_res<Scalar, Device_, Layout>(prev_layer);

        DeviceTensor<Scalar, 4, Device_, Layout> dilated = dilate_tensor(kernel, dilation);
        DeviceTensor<Scalar, 2, Device_, Layout> unf_dilated = unfold_kernel(dilated);

        // transposed convolution: kernel.T * x_col
        ProductDims prod_dims = { IndexPair<int>(0, 0) };
        DeviceTensor<Scalar, 2, Device_, Layout>  X_col(unf_dilated.dimension(1), inp_reshaped.dimension(1));
        X_col.view() = unf_dilated->contract(*inp_reshaped, prod_dims);

        // re-format into the image of output dimensions
        col2im(X_col, layer_output, *params, true);
#ifdef __CUDACC__
      }
#endif

      // one bias per filter
      layer_output.view() += bias->reshape(array<Index, 4>{ 1, kernel.dimension(1), 1, 1 }).broadcast(bias_broadcast);

    }

    void backward(PtrTensorAdapter<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, 4, Device_, Layout> prev_layer(prev_layer_any);
      DeviceTensor<Scalar, 4, Device_, Layout> next_layer_grad(next_layer_grad_any);
      //bias
      loss_by_bias_derivative.view() = next_layer_grad->sum(array<Index, 3>{0, 2, 3});

#ifdef __CUDACC__
      if (is_cudnn) {

        // data backwards
        checkCudnnErrors(cudnnConvolutionForward(CudnnWorkspace::cudnn(), &(CudnnWorkspace::one), cudnn_workspace->input_desc, next_layer_grad->data(),
          cudnn_workspace->filter_desc, kernel->data(), cudnn_workspace->conv_desc, cudnn_workspace->conv_fwd_algo, CudnnWorkspace::workspace(), CudnnWorkspace::workspace_size,
          &(CudnnWorkspace::zero), cudnn_workspace->output_desc, dX->data()));

        // weights backwards
        checkCudnnErrors(
          cudnnConvolutionBackwardFilter(CudnnWorkspace::cudnn(), &(CudnnWorkspace::one), cudnn_workspace->input_desc, next_layer_grad->data(),
            cudnn_workspace->output_desc, prev_layer->data(),
            cudnn_workspace->conv_desc, cudnn_workspace->conv_bwd_filter_algo, 
            CudnnWorkspace::workspace(), CudnnWorkspace::workspace_size, 
            &(CudnnWorkspace::zero), cudnn_workspace->filter_desc, dW->data()));

        return;
      }
#endif // __CUDACC__

      // dX: kernel.T.T * dout which is just a regular convolution
      dX = convolve<Scalar, 4, Layout, Device_>(next_layer_grad, kernel, *params);

      DeviceTensor<Scalar, 2, Device_, Layout> dout = im2col<Scalar, 4, Device_, Layout>(next_layer_grad, *params);

      // dW: (dout * x_col.T.T)
      DeviceTensor<Scalar, 2, Device_, Layout> inp_reshaped = unfold_conv_res<Scalar, Device_, Layout>(prev_layer);
      ProductDims prod_dims = { IndexPair<int>(1, 1) };
      DeviceTensor<Scalar, 2, Device_, Layout>  dW_col(inp_reshaped.dimension(0), dout.dimension(0));
      dW_col.view() = inp_reshaped->contract(*dout, prod_dims);

      dW = fold_kernel(dW_col, kernel.dimensions());

    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return dX.raw();
    }

    // feed to optimizer
    PtrTensorAdapter<Scalar, Device_> get_loss_by_weights_derivative() override {
      return dW.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_weights() override {
      return kernel.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_bias() override {
      return bias.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_bias_derivative() override {
      return loss_by_bias_derivative.raw();
    }

    void set_weights(PtrTensorAdapter<Scalar, Device_>& v) override {
      kernel = DeviceTensor<Scalar, 4, Device_, Layout>(v);
    }

    void set_bias(PtrTensorAdapter<Scalar, Device_>& v) override {
      bias = DeviceTensor<Scalar, 1, Device_, Layout>(v);
    }

    inline void set_cudnn(bool _is_cudnn) {
      assert(!_is_cudnn || Layout & RowMajor);
      is_cudnn = _is_cudnn;
    }

    // add ONNX node corresponding to this layer
    // returns the name of the output in the serialized file
    const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override {

      // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#ConvTranspose
      // 1. Save initializers (raw data in row major format)
      kernel.save_onnx_initializer(model);
      bias.save_onnx_initializer(model);

      // 2. add ONNX node with its inputs, outputs, and names
      std::vector<std::string> names{ input_name, kernel.get_onnx_input_name(model), bias.get_onnx_input_name(model) };
      onnx::NodeProto* node = model.add_graph_node(get_op_name(), names);

      // single output
      const std::string out_name = node->output().Get(0);

      // 3. create attributes
      params->create_onnx_attributes(node);

      // return output to pass as input to next node in graph
      return out_name;
    }

    // in the order they appear in the ONNX file
    // in the order they appear in the ONNX file
    inline void load_onnx_data(EigenModel& model, std::vector<std::string>& inputs) override {

      std::vector<std::vector<Index>> dimensions;
      std::vector<onnx::TensorProto> values;

      std::tie(values, dimensions) = model.get_input_data_and_dimensions<Scalar>(inputs);

      kernel.set_from_host(model.get_input_data<Scalar>(values[0]), vec2dims<4>(dimensions[0]));
      bias.set_from_host(model.get_input_data<Scalar>(values[1]), vec2dims<1>(dimensions[1]));

      // inputs are stored weights only, exclude the actual input tensor
      kernel.set_node_input_name(inputs[1]);
      bias.set_node_input_name(inputs[2]);
    }

    const std::vector<Index> onnx_out_dims() override {
      return layer_output.vec_dims();
    }

  private:
    DeviceTensor<Scalar, 4, Device_, Layout> kernel, layer_output, dX, dW;
    DeviceTensor<Scalar, 1, Device_, Layout> bias, loss_by_bias_derivative;

    const int stride, dilation;
    const Padding2D padding;
    array<Index, 4> bias_broadcast;
    std::shared_ptr<ConvolutionParams<4>> params;

#ifdef __CUDACC__
    std::shared_ptr<CudnnWorkspace> cudnn_workspace;
#endif
  };
}