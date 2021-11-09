cmake_minimum_required(VERSION 3.20.2)
project(cifar LANGUAGES CXX CUDA)

#set( CMAKE_TOOLCHAIN_FILE "C:/git/NN/vcpkg/scripts/buildsystems/vcpkg.cmake" )

find_package(OpenCV REQUIRED)

# CIFAR10 reader 
# Clone https://github.com/wichtounet/cifar-10
# and point cmake to the location of cifar-10-batches-bin

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Protobuf
# From stackoverflow: https://stackoverflow.com/questions/53651181/cmake-find-protobuf-package-in-custom-directory
include(${Protobuf_ROOT}/protobuf-config.cmake)
include(${Protobuf_ROOT}/protobuf-module.cmake)
include(${Protobuf_ROOT}/protobuf-options.cmake)
include(${Protobuf_ROOT}/protobuf-targets.cmake)

find_package(Protobuf REQUIRED HINTS ${Protobuf_ROOT})

include_directories(${Protobuf_INCLUDE_DIRS})

get_filename_component(EIGEN3_INCLUDE_DIRS "../../core/eigen" ABSOLUTE)
get_filename_component(EIGENSINN_INCLUDE_DIR "../../core/nn/src" ABSOLUTE)
get_filename_component(CUR_INCLUDE_DIR "include" ABSOLUTE)

# download CIFAR10 dataset as described in the repo above.
# untar to the current directory
get_filename_component(CIFAR_DATA_DIR "cifar-10-batches-bin" ABSOLUTE)

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen path: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "EigenSiNN path: ${EIGENSINN_INCLUDE_DIR}")
message(STATUS "OpenCV path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Local includes:  ${CUR_INCLUDE_DIR}")
message(STATUS "CIFAR10 location: ${CIFAR_DATA_DIR}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

set(APP_EXE simple_cifar)

set(WITH_GPU false)
set(COL_MAJOR false)

find_package(Boost REQUIRED COMPONENTS graph system serialization)
# discover sources
file(GLOB_RECURSE APP_SOURCES LIST_DIRECTORIES false "src/*.cpp")

if(WITH_GPU)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA Arches" FORCE)
    
    file(GLOB_RECURSE APP_SOURCES LIST_DIRECTORIES false "src/*.cu")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr  --default-stream per-thread")

endif()

set(APP_SOURCES ${APP_SOURCES} "${EIGENSINN_INCLUDE_DIR}/onnx/onnx.proto3.pb.cc")
message(STATUS "Sources: ${APP_SOURCES}")

# build app exe
add_executable(${APP_EXE} ${APP_SOURCES})

set(COMPILE_DEFS CIFAR_DATA_LOCATION="${CIFAR_DATA_DIR}" EIGEN_USE_THREADS)

if(COL_MAJOR)
    set(COMPILE_DEFS ${COMPILE_DEFS} COLMAJOR)
endif()

if(WITH_GPU)
    set_property(TARGET ${APP_EXE} 
             PROPERTY CMAKE_CUDA_ARCHITECTURES $CMAKE_CUDA_ARCHITECTURES)
    set(COMPILE_DEFS ${COMPILE_DEFS} EIGEN_USE_GPU EIGEN_HAS_C99_MATH)
endif()

target_compile_definitions(${APP_EXE} PRIVATE ${COMPILE_DEFS})
target_include_directories(${APP_EXE} PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIGENSINN_INCLUDE_DIR} ${CUR_INCLUDE_DIR} ${Protobuf_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(${APP_EXE} PUBLIC ${OpenCV_LIBS} ${Protobuf_LIBRARIES}  ${Boost_LIBRARIES} cudnn)
