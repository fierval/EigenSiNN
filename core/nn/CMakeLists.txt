set(OpenCV_DIR $ENV{OPENCV_INSTALL})

find_package(OpenCV)

# boost graph 
find_package(Boost)
if(Boost_FOUND)
    set(BOOST_INCLUDE ${Boost_INCLUDE_DIRS})
    set(BOOST_LIBS ${Boost_LIBRARIES})
endif()

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

get_filename_component(EIGEN_PATH "../eigen" ABSOLUTE)
message(STATUS "Eigen path: ${EIGEN_PATH}")

set(TEST_EXE eigentest)

# discover test sources
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false "test/*.cpp")
message(STATUS "Sources: ${TEST_SOURCES}")

# build test exe
add_executable(${TEST_EXE} ${TEST_SOURCES} "src/losses/loss_base.hpp")
target_include_directories(${TEST_EXE} PRIVATE "src" ${EIGEN_PATH} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(${TEST_EXE} PUBLIC gtest_main ${OpenCV_LIBS} ${Boost_LIBRARIES})

# gtest specific
set(gtest_force_shared_crt ON)
gtest_discover_tests(${TEST_EXE})