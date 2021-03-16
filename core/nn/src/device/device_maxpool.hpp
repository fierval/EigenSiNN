#pragma once

#include "device_helpers.hpp"
#include <limits>

namespace EigenSinn {

#ifdef __CUDACC__

  template<typename Scalar, int Layout = ColMajor>
  __global__ void maxpool_forward_kernel(long h_offset, long w_offset, long kernel_height, long kernel_width, long stride, long dilation, TensorView<Scalar, 4, Layout> input, TensorView<Scalar, 4, Layout> layer_output, TensorView<Index, 4, Layout> mask) {

    DSizes<long, 4> out_dims = dimensions_cast<long>(layer_output.dimensions());

    long out_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (out_index >= out_dims.TotalSize()) {
      return;
    }

    DSizes<long, 4> dims = dimensions_cast<long>(input.dimensions());

    DSizes<long, 4> offsets = from_flat_dim<long, 4, Layout>(out_dims, out_index);

    Scalar max_val = std::numeric_limits<Scalar>::lowest();
    long max_idx = -1;

    for (long kernel_h = 0; kernel_h < kernel_height; kernel_h += dilation) {
      for (long kernel_w = 0; kernel_w < kernel_width; kernel_w += dilation) {
        long cur_h = h_offset + offsets[2] * stride + kernel_h;
        long cur_w = w_offset + offsets[3] * stride + kernel_w;

        Scalar val;

        if (cur_h >= 0 && cur_h < dims[2] && cur_w >= 0 && cur_w < dims[3]) {
          long input_idx = to_flat_dim<long, 4, Layout>(dims, { offsets[0], offsets[1], cur_h, cur_w });

          val = input.data()[input_idx];
          max_idx = (val > max_val) ? input_idx : max_idx;
          max_val = (val > max_val) ? val : max_val;
        }
      }
    }
    layer_output.data()[out_index] = max_val;
    mask.data()[out_index] = max_idx;

  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void maxpool_backward_kernel(long h_offset, long w_offset, long kernel_h, long kernel_w, int stride, int dilation, TensorView<Scalar, 4, Layout> next_layer_grad, TensorView<Index, 4, Layout> mask, TensorView<Scalar, 4, Layout> layer_gradient)
  {

    // same as input dimensions
    DSizes<long, 4> dims = dimensions_cast<long>(layer_gradient.dimensions());
    long in_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (in_idx >= dims.TotalSize()) {
      return;
    }

    long area_w = (kernel_w - 1) / stride;
    long area_h = (kernel_h - 1) / stride;

    DSizes<long, 4> out_dims = dimensions_cast<long>(mask.dimensions());
    DSizes<long, 4> in_offsets = from_flat_dim<long, 4, Layout>(dims, in_idx);
    
    float d = 0;
    for (long cur_inp_h = -area_h; cur_inp_h < area_h + 1; cur_inp_h += dilation) {
      for (long cur_inp_w = -area_w; cur_inp_w < area_w + 1; cur_inp_w += dilation) {

        long out_w = (in_offsets[3] - w_offset) / stride + cur_inp_w;
        long out_h = (in_offsets[2] - h_offset) / stride + cur_inp_h;
        long out_index = to_flat_dim<long, 4, Layout>(out_dims, { in_offsets[0], in_offsets[1], out_h, out_w });

          if (out_w >= 0 && out_w < out_dims[3] && out_h >= 0 && out_h < out_dims[2] && mask.data()[out_index] == in_idx) {
          
            d += next_layer_grad.data()[out_index];
          }
      }
    }
    layer_gradient.data()[in_idx] += d;
  }


  template<typename Scalar, int Layout = ColMajor>
  __global__ void maxpool_set_values_kernel4d(
    TensorView<Scalar, 4, Layout> output, TensorView<Index, 4, Layout> mask, TensorView<Tuple<Index, Scalar>, 2, Layout> local_pool, 
      dim3 in_size, dim3 output_starts) {

    int batch = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = threadIdx.y + blockIdx.y * blockDim.y;
    
    int channels = in_size.y;
    int batches = in_size.x;

    if (batch < batches && channel < channels) {

      // TODO: for some reason creating Index-type (long long) arrays crashes the kernel
      // so convering everything to "long"
      auto dims = mask.dimensions();
      auto out_dims = array<long, 4>{(long)dims[0], (long)dims[1], (long)dims[2], (long)dims[3]};

      array<long, 2> local_dims{ batches, channels };

      auto idx_output = to_flat_dim<long, 4, Layout>(out_dims, { batch, channel, (long)output_starts.y, (long)output_starts.x });
      auto idx_tuple = to_flat_dim<long, 2, Layout>(local_dims, { batch, channel });

      mask.data()[idx_output] = local_pool.data()[idx_tuple].first;
      output.data()[idx_output] = local_pool.data()[idx_tuple].second;
    }

  }

  template<typename Scalar, int Layout = ColMajor>
  __global__ void maxpool_dinput_kernel4d(
    TensorView<Scalar, 4, Layout> output, TensorView<Scalar, 4, Layout> dout, TensorView<Index, 4, Layout> mask
    , long batches, long channels, dim3 grad_starts, dim3 extents, dim3 output_pos) {

    int batch = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = threadIdx.y + blockIdx.y * blockDim.y;

    if (batch < batches && channel < channels) {

      array<long, 4> pool_window_dims{ batches, channels, (long)extents.y, (long)extents.x };

      // TODO: for some reason creating Index-type (long long) arrays crashes the kernel
      // so convering everything to "long"
      auto dims = mask.dimensions();
      auto mask_dims = array<long, 4>{(long)dims[0], (long)dims[1], (long)dims[2], (long)dims[3]};

      dims = output.dimensions();
      auto out_dims = array<long, 4>{(long)dims[0], (long)dims[1], (long)dims[2], (long)dims[3]};

      auto idx = to_flat_dim<long, 4, Layout>(mask_dims, { batch, channel, (long)grad_starts.y, (long)grad_starts.x });
      auto idx_flat = mask.data()[idx];

      auto unrolled_dim = from_flat_dim<long, 4, ColMajor>(pool_window_dims, idx_flat);

      long idx_output = to_flat_dim<long, 4, Layout>(out_dims, { batch, channel, (long)output_pos.y + unrolled_dim[2], (long)output_pos.x + unrolled_dim[3] });
      long idx_grad = to_flat_dim<long, 4, Layout>(mask_dims, { batch, channel, (long)grad_starts.y, (long)grad_starts.x });

      output.data()[idx_output] += dout.data()[idx_grad];
    }
  }
#endif  
} // namespace EigenSinn