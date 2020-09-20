set(OpenCV_DIR $ENV{OPENCV_INSTALL})

find_package(OpenCV CONFIG REQUIRED)

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)
message(STATUS "Eigen path: ${EIGEN_PATH}")

set(TEST_EXE eigentest)

# discover test sources
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false "test/*.cpp" "test/*.cu")
message(STATUS "Sources: ${TEST_SOURCES}")
message(STATUS "CUDA Include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
find_package(Boost REQUIRED COMPONENTS graph)

# build test exe
add_executable(${TEST_EXE} ${TEST_SOURCES})

set_property(TARGET ${TEST_EXE} PROPERTY CUDA_STANDARD 14)

set_property(TARGET ${TEST_EXE} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# allow threadpool device
target_compile_definitions(${TEST_EXE} PRIVATE EIGEN_USE_THREADS)

target_include_directories(${TEST_EXE} PRIVATE "src" ${EIGEN_PATH} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${TEST_EXE} PUBLIC gtest_main ${OpenCV_LIBS} )

# gtest specific
set(gtest_force_shared_crt ON)
gtest_discover_tests(${TEST_EXE})