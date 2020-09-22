cmake_minimum_required(VERSION 3.17)

# set the project name
project(EgenSiNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO: This is a hack until 
# OpenCV supports CMAKE_CUDA_STANDARD 17
# We manually add the --std=c++17 flag 
# for CUDA compilation
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED False)

include_directories("eigen")

enable_testing()
include(GoogleTest)
add_subdirectory(googletest)
add_subdirectory(nn)
