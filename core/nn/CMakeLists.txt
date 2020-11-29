message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)
message(STATUS "Eigen path: ${EIGEN_PATH}")

set(TEST_EXE eigentest)

# discover test sources
#file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false "test/*.cpp" "test/*.cu")
set(TEST_SOURCES "test/device_tensor_test.cpp" "test/gpu_device_tensor_test.cu" "test/linearlayer_test.cpp" "test/linear_gpu_test")
#set(TEST_SOURCES "test/device_tensor_test.cpp" "test/gpu_device_tensor_test.cu")

message(STATUS "Sources: ${TEST_SOURCES}")
message(STATUS "CUDA Include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
find_package(Boost REQUIRED COMPONENTS graph)

# build test exe
add_executable(${TEST_EXE} ${TEST_SOURCES})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")

set_property(TARGET ${TEST_EXE} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET ${TEST_EXE} 
             PROPERTY CMAKE_CUDA_ARCHITECTURES 70)

# allow threadpool device
target_compile_definitions(${TEST_EXE} PRIVATE EIGEN_USE_THREADS EIGEN_USE_GPU EIGEN_HAS_C99_MATH)

target_include_directories(${TEST_EXE} PRIVATE "src" ${EIGEN_PATH} ${Boost_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${TEST_EXE} PUBLIC gtest_main )

# gtest specific
set(gtest_force_shared_crt ON)
gtest_discover_tests(${TEST_EXE})