# CMake Tools
# Author: Kai Jin
# Updated: 11/01/2018
# mail: <atranitell@gmail.com>
cmake_minimum_required(VERSION 3.4)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0010 NEW)
project(ocl_benchmarks VERSION 0.0.1)

# this should be modify
set(THIRD_PARTY_DIR $ENV{TEA_THIRD_PARTY})

# configure platform compile options
set(CMAKE_BUILD_TYPE "Release")
set(BUILD_SHARED_LIBS TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
elseif(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-result -std=c++14 -O3 -fPIC")
  link_libraries(pthread)
elseif(APPLE)
  message(FATAL "[ERR] Configuration has not been implemented.")
endif()

# include
include(${CMAKE_SOURCE_DIR}/cmake/find_opencl.cmake)
include_directories(${CMAKE_SOURCE_DIR})
# src
add_executable(ocl_benchmarks 
  "${CMAKE_SOURCE_DIR}/benchmarks/main.cc"
  "${CMAKE_SOURCE_DIR}/benchmarks/ocl_helper.cc")