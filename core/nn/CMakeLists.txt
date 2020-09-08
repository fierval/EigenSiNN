set(OpenCV_DIR $ENV{OPENCV_INSTALL})

find_package(OpenCV CONFIG REQUIRED)

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)
message(STATUS "Eigen path: ${EIGEN_PATH}")

set(TEST_EXE eigentest)

# discover test sources
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false "test/*.cpp" "test/*.cu")
message(STATUS "Sources: ${TEST_SOURCES}")

find_package(Boost REQUIRED COMPONENTS graph)

# build test exe
add_executable(${TEST_EXE} ${TEST_SOURCES})

# allow threadpool device
target_compile_definitions(${TEST_EXE} PRIVATE EIGEN_USE_THREADS)

target_include_directories(${TEST_EXE} PRIVATE "src" ${EIGEN_PATH} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(${TEST_EXE} PUBLIC gtest_main ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CUDA_LIBRARIES})

# gtest specific
set(gtest_force_shared_crt ON)
gtest_discover_tests(${TEST_EXE})