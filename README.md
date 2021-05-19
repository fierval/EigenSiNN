# EigenSiNN

Eigen based cross-platform header-only C++ library for deep learning with emphasis on computer vision, written to help educate myself on details of neural nets.

[Eigen](https://gitlab.com/libeigen/eigen) is used for all of the CPU and basics of the GPU implementation. 

[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) v8.1.1 is available for Convolution, Transposed Convolution, Max Pool, ReLU, Sigmoid, and Tanh.

## API

### Supported Layers

* Fully connected
* Convolutional (except grouped convolutions)
* Transposed Convolutional
* Fully Connected
* Max Pooling
* Batch Normalization
* Dropout

### Activations

* Sigmoid
* Tanh
* Leaky ReLU
* ReLU
* Softmax

### Losses

* MSE
* Cross Entropy

### Optimizers 

* SGD
* Adam

## Installation

### Prerequisits

* C++ 17
* CUDA 11.2 + cuDNN v8.1.1 for training/inferencing on GPU

### To Build Tests

* vcpkg Package Manager
* Boost: not included in the library, a test builds/runs boost graph for future development

## Tests

Tests together with the CIFAR-10 classifier sample provide usage examples. [CMakeLists.txt](core/nn/CMakeLists.txt) shows how to build an app with `EigenSiNN` library. In particular:

```cmake
target_compile_definitions(${TEST_EXE} PRIVATE EIGEN_USE_THREADS EIGEN_USE_GPU EIGEN_HAS_C99_MATH)
```
need to be defined.

## Cifar-10 Classification Example Network

To build the sample CIFAR-10 [classifier](networks):

1. Build and install [OpenCV 4.5.2](https://github.com/opencv/opencv) with CUDA
1. in [CMakeLists.txt](networks/cifar/CMakeLists.txt), set ``WITH_GPU`` to build with GPU support. in [main.cpp](networks/cifar/src/main.cpp) or [main.cu](networks/cifar/src/main.cu), depending on the CPU/GPU build, set:

 ```cpp
 bool explore_dataset = true;
 ```

  at the top of the file to display images from CIFAR-10 dataset before the training runs. Hit any key to navigate the dataset forward, ESC to exit and start training.