
add_library(dummy src/dummy.cpp)

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

add_executable(eigentest test/dummytest.cpp)
get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)

message(STATUS "Eigen path: ${EIGEN_PATH}")

target_include_directories(eigentest PRIVATE "src" ${EIGEN_PATH})
target_link_libraries(eigentest dummy)