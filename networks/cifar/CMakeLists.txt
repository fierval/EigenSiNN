cmake_minimum_required(VERSION 3.17)
project(cifar LANGUAGES CXX CUDA)

#set( CMAKE_TOOLCHAIN_FILE "C:/git/NN/vcpkg/scripts/buildsystems/vcpkg.cmake" )

find_package(OpenCV REQUIRED)

# CIFAR10 reader 
# Clone https://github.com/wichtounet/cifar-10
# and point cmake to the repo root

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA Arches" FORCE)

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

set(WITH_CPU false)

# discover sources
if(WITH_CPU)
    file(GLOB_RECURSE APP_SOURCES LIST_DIRECTORIES false "src/*.cpp")
else()
    file(GLOB_RECURSE APP_SOURCES LIST_DIRECTORIES false "src/*.cu")
endif()

message(STATUS "Sources: ${APP_SOURCES}")

# build app exe
add_executable(${APP_EXE} ${APP_SOURCES})

set_property(TARGET ${APP_EXE} 
             PROPERTY CMAKE_CUDA_ARCHITECTURES $CMAKE_CUDA_ARCHITECTURES)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

target_include_directories(${APP_EXE} PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIGENSINN_INCLUDE_DIR} ${CUR_INCLUDE_DIR})
target_compile_definitions(${APP_EXE} PRIVATE CIFAR_DATA_LOCATION="${CIFAR_DATA_DIR}" EIGEN_USE_THREADS EIGEN_USE_GPU EIGEN_HAS_C99_MATH)
target_link_libraries(${APP_EXE} PUBLIC ${OpenCV_LIBS})
