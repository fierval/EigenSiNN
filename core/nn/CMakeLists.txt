message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)
message(STATUS "Eigen path: ${EIGEN_PATH}")

set(TEST_EXE eigentest)

# discover test sources
 file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false "test/*.cpp" "test/*.cu" "src/onnx/*.cc")
set(TEST_SOURCES 
    "src/onnx/onnx.proto3.pb.cc"
    "test/onnx_test.cpp"
#    "test/batchnorm4d_test.cpp"
    "test/network_graph_test.cpp"
#    "test/sigmoid_gpu_test.cu"
#    "test/sigmoid4d_gpu_test.cu"
#    "test/sigmoid_test.cpp"
#     "test/maxpooling4d_gpu_test.cu"
#    "test/cudnn_test.cu"
#    "test/convlayer_test.cpp"
#    "test/convlayer_cudnn_test.cu"
#    "test/convlayer_gpu_test.cu"
#    "test/memory_test_gpu.cu"
#    "test/convlayer_trans_test.cpp"
#    "test/convlayer_trans_gpu_test.cu"
#    "test/convlayer_trans_cudnn_test.cu"
#    "test/maxpooling4d_test.cpp"
#    "test/maxpooling4d_rowmajor_test.cpp"
#    "test/maxpooling4d_gpu_test.cu"
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

target_include_directories(${TEST_EXE} PRIVATE "src" ${EIGEN_PATH} ${Boost_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Protobuf_INCLUDE_DIRS})

target_link_libraries(${TEST_EXE} PUBLIC gtest_main ${Boost_LIBRARIES} ${Protobuf_LIBRARIES})
# target_link_libraries(${TEST_EXE} PUBLIC gtest_main ${Boost_LIBRARIES} ${Protobuf_LIBRARIES} cudnn)

# gtest specific
set(gtest_force_shared_crt ON)
gtest_discover_tests(${TEST_EXE})