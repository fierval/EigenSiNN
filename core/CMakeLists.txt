cmake_minimum_required(VERSION 3.17)

# set the project name
project(EgenSiNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("eigen")

find_package(CUDA)

if (CUDA_FOUND)
  message(STATUS "found cuda")
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

enable_testing()
include(GoogleTest)
add_subdirectory(googletest)
add_subdirectory(nn)
