cmake_minimum_required(VERSION 3.10)

# set the project name
project(EgenSiNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("eigen")
add_subdirectory(nn)
