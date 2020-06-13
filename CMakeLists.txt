cmake_minimum_required(VERSION 3.17)

# set the project name
project(EgenSiNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

include_directories("eigen")
add_subdirectory(nn)
