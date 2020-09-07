#include <unsupported/Eigen/CXX11/Tensor>
#include <gtest/gtest.h>

using namespace Eigen;

namespace EigenTest {
  TEST(CUDA, Reduction)
  {
    Tensor<float, 4, ColMajor, int> in1(72, 53, 97, 113);
    Tensor<float, 2, ColMajor, int> out(72, 97);
    in1.setRandom();

    std::size_t in1_bytes = in1.size() * sizeof(float);
    std::size_t out_bytes = out.size() * sizeof(float);

    float* d_in1;
    float* d_out;
    cudaMalloc((void**)(&d_in1), in1_bytes);
    cudaMalloc((void**)(&d_out), out_bytes);

    cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);

    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice gpu_device(&stream);

    Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor, int> > gpu_in1(d_in1, 72, 53, 97, 113);
    Eigen::TensorMap<Eigen::Tensor<float, 2, ColMajor, int> > gpu_out(d_out, 72, 97);

    array<Eigen::DenseIndex, 2> reduction_axis;
    reduction_axis[0] = 1;
    reduction_axis[1] = 3;

    gpu_out.device(gpu_device) = gpu_in1.maximum(reduction_axis);

    assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

    cudaFree(d_in1);
    cudaFree(d_out);
  }
}