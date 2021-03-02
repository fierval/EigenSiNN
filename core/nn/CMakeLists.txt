message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)
message(STATUS "Eigen path: ${EIGEN_PATH}")

set(TEST_EXE eigentest)

# discover test sources
#file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false "test/*.cpp" "test/*.cu")
set(TEST_SOURCES "test/device_tensor_test.cpp" 
                "test/device_tensor_gpu_test.cu" 
                "test/convlayer_test.cpp" 
                "test/convlayer_gpu_test.cu" 
                "test/linear_test.cpp" 
                "test/linear_gpu_test.cu"
                "test/convlayer_trans_gpu_test.cu"
                "test/convlayer_trans_test.cpp"
                "test/batchnorm_test.cpp"
                "test/batchnorm4d_test.cpp"
                "test/dropout_test.cpp"
                "test/flatten_test.cpp"
                "test/maxpooling2d_test.cpp"
                "test/maxpooling4d_test.cpp"
                "test/maxpooling2d_gpu_test.cu"
                "test/maxpooling4d_gpu_test.cu"
                "test/relu_test.cpp"
                "test/relu_gpu_test.cu"
                "test/sigmoid_test.cpp"
                "test/softmax_test.cpp"
                "test/tanh_test.cpp"
                "test/losses_test.cpp"
                "test/sgd_test.cpp"
                )

message(STATUS "Sources: ${TEST_SOURCES}")
message(STATUS "CUDA Include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
find_package(Boost REQUIRED COMPONENTS graph system serialization)

# build test exe
add_executable(${TEST_EXE} ${TEST_SOURCES})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")

set_property(TARGET ${TEST_EXE} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA Arches" FORCE)

set_property(TARGET ${TEST_EXE} 
             PROPERTY CMAKE_CUDA_ARCHITECTURES $CMAKE_CUDA_ARCHITECTURES)

# allow threadpool and GPU devices
target_compile_definitions(${TEST_EXE} PRIVATE EIGEN_USE_THREADS EIGEN_USE_GPU EIGEN_HAS_C99_MATH)

target_include_directories(${TEST_EXE} PRIVATE "src" ${EIGEN_PATH} ${Boost_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${TEST_EXE} PUBLIC gtest_main ${Boost_LIBRARIES} )

# gtest specific
set(gtest_force_shared_crt ON)
gtest_discover_tests(${TEST_EXE})