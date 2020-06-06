
add_library(dummy src/dummy.cpp)

message(STATUS "Using C++${CMAKE_CXX_STANDARD}")

add_executable(eigentest test/dummytest.cpp)

target_include_directories(eigentest PRIVATE "src")
target_link_libraries(eigentest dummy)