
add_library(eigensinn src/dummy.cpp)

add_executable(eigentest test/dummytest.cpp)

target_include_directories(eigentest PRIVATE "src")
target_link_libraries(eigentest eigensinn)