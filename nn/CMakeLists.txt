set(OpenCV_DIR $ENV{OPENCV_INSTALL})

find_package( OpenCV REQUIRED )

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)
message(STATUS "Eigen path: ${EIGEN_PATH}")

set(TEST_EXE eigentest)

# discover test sources
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false "test/*.cpp")
message(STATUS "Sources: ${TEST_SOURCES}")

# build test exe
add_executable(${TEST_EXE} ${TEST_SOURCES})
target_include_directories(${TEST_EXE} PRIVATE "src" ${EIGEN_PATH} ${OPENCV_INCLUDE})
target_link_libraries(${TEST_EXE} PUBLIC gtest_main)

# gtest specific
set(gtest_force_shared_crt ON)
gtest_discover_tests(${TEST_EXE})