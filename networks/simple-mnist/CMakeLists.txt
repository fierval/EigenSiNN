cmake_minimum_required(VERSION 3.17)

#set( CMAKE_TOOLCHAIN_FILE "C:/git/NN/vcpkg/scripts/buildsystems/vcpkg.cmake" )

# set the project name
project(EgenSiNN LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

get_filename_component(EIGENSINN_INCLUDE_DIR "../../core/nn/src" ABSOLUTE)

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen path: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "EigenSiNN path: ${EIGENSINN_INCLUDE_DIR}")
message(STATUS "OpenCV path: ${OpenCV_INCLUDE_DIRS}")

set(APP_EXE mnist_class)

# discover sources
file(GLOB_RECURSE APP_SOURCES LIST_DIRECTORIES false "src/*.cpp")
message(STATUS "Sources: ${APP_SOURCES}")

# build app exe
add_executable(${APP_EXE} ${APP_SOURCES})

target_include_directories(${APP_EXE} PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIGENSINN_INCLUDE_DIR})
target_link_libraries(${APP_EXE} PUBLIC ${OpenCV_LIBS})
