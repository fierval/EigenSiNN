
add_library(dummy src/dummy.cpp)

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)
set(This eigentest)
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false "test/*.cpp")

message(STATUS "Sources: ${TEST_SOURCES}")
add_executable(${This} ${TEST_SOURCES})

message(STATUS "Eigen path: ${EIGEN_PATH}")

target_include_directories(${This} PRIVATE "src" ${EIGEN_PATH})
target_link_libraries(${This} PUBLIC gtest_main)

set(gtest_force_shared_crt ON)
gtest_discover_tests(${This})