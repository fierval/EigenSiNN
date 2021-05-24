cmake_minimum_required(VERSION 3.20.2)

# set the project name
project(EgenSiNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)

# Protobuf
# From stackoverflow: https://stackoverflow.com/questions/53651181/cmake-find-protobuf-package-in-custom-directory
include(${Protobuf_ROOT}/protobuf-config.cmake)
include(${Protobuf_ROOT}/protobuf-module.cmake)
include(${Protobuf_ROOT}/protobuf-options.cmake)
include(${Protobuf_ROOT}/protobuf-targets.cmake)

find_package(Protobuf REQUIRED HINTS ${Protobuf_ROOT})

include_directories(${Protobuf_INCLUDE_DIRS})

message(STATUS "Protobuf libraries: ${Protobuf_LIBRARIES}")
message(STATUS "Protobuf includes: ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Protobuf version: ${Protobuf_VERSION}")
include_directories("eigen")
enable_testing()
include(GoogleTest)
add_subdirectory(nn)
