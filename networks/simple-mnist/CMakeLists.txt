project(simple-mnist)
cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# MNIST reader package
# Clone https://github.com/wichtounet/mnist
# and point cmake to the repo root
find_package(MNIST REQUIRED PATHS ../../../mnist)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

get_filename_component(EIGENSINN_INCLUDE_DIR "../../core/nn/src" ABSOLUTE)
get_filename_component(CUR_INCLUDE_DIR "include" ABSOLUTE)

message(STATUS "MNIST data dir: ${MNIST_DATA_DIR}")
set(APP_EXE simple_mnist)

# discover sources
file(GLOB_RECURSE APP_SOURCES LIST_DIRECTORIES false "src/*.cpp")
message(STATUS "Sources: ${APP_SOURCES}")
message(STATUS "Using C++${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen path: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "EigenSiNN path: ${EIGENSINN_INCLUDE_DIR}")
message(STATUS "OpenCV path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Local includes:  ${CUR_INCLUDE_DIR}")

# build app exe
add_executable(${APP_EXE} ${APP_SOURCES})

target_compile_definitions(${APP_EXE} PRIVATE EIGEN_USE_THREADS)

target_include_directories(${APP_EXE} PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIGENSINN_INCLUDE_DIR} ${MNIST_INCLUDE_DIR} ${CUR_INCLUDE_DIR})
target_compile_definitions(${APP_EXE} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
target_link_libraries(${APP_EXE} PUBLIC ${OpenCV_LIBS})
