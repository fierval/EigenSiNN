cmake_minimum_required(VERSION 3.17)

#set( CMAKE_TOOLCHAIN_FILE "C:/git/NN/vcpkg/scripts/buildsystems/vcpkg.cmake" )

# set the project name
project(simple-mnist LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# MNIST reader package
find_package(MNIST REQUIRED PATHS ../../../mnist)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

get_filename_component(EIGENSINN_INCLUDE_DIR "../../core/nn/src" ABSOLUTE)

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen path: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "EigenSiNN path: ${EIGENSINN_INCLUDE_DIR}")
message(STATUS "OpenCV path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "MNIST data reader: ${MNIST_INCLUDE_DIR}")

set(APP_EXE simple_mnist)

# discover sources
file(GLOB_RECURSE APP_SOURCES LIST_DIRECTORIES false "src/*.cpp")
message(STATUS "Sources: ${APP_SOURCES}")

# build app exe
add_executable(${APP_EXE} ${APP_SOURCES})

target_include_directories(${APP_EXE} PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIGENSINN_INCLUDE_DIR} ${MNIST_INCLUDE_DIR})
target_compile_definitions(${APP_EXE} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
target_link_libraries(${APP_EXE} PUBLIC ${OpenCV_LIBS})
