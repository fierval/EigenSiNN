cmake_minimum_required(VERSION 3.17)

# set the project name
project(EgenSiNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CUDA_STANDARD 14)

include_directories("eigen")

enable_testing()
include(GoogleTest)
add_subdirectory(googletest)
add_subdirectory(nn)
