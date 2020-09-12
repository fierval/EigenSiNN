cmake_minimum_required(VERSION 3.17)
project(cifar LANGUAGES CXX)

#set( CMAKE_TOOLCHAIN_FILE "C:/git/NN/vcpkg/scripts/buildsystems/vcpkg.cmake" )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# CIFAR10 reader 
# Clone https://github.com/wichtounet/cifar-10
# and point cmake to the repo root

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
set(APP_EXE simple_cifar)

# discover sources
file(GLOB_RECURSE APP_SOURCES LIST_DIRECTORIES false "src/*.cpp")
message(STATUS "Sources: ${APP_SOURCES}")

# build app exe
add_executable(${APP_EXE} ${APP_SOURCES})

target_include_directories(${APP_EXE} PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIGENSINN_INCLUDE_DIR} ${CUR_INCLUDE_DIR})
target_compile_definitions(${APP_EXE} PRIVATE CIFAR_DATA_LOCATION="${CIFAR_DATA_DIR}" EIGEN_USE_THREADS)
target_link_libraries(${APP_EXE} PUBLIC ${OpenCV_LIBS})
