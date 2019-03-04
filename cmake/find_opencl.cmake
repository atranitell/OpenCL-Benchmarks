# Find OPENCL libraries
# Author: Kai Jin
# Updated: 09/18/2018
# mail: <atranitell@gmail.com>

# OPENCL_FOUND
# OPENCL_INCLUDE_DIR
# OPENCL_LIBRARIES

if(NOT OPENCL_IMPORT STREQUAL "TRUE")
  if(WIN32)
    find_file(OPENCL_INCLUDE_DIR opencl/include PATHS ${THIRD_PARTY_DIR})
    find_file(OPENCL_BIN_DIR opencl/bin PATHS ${THIRD_PARTY_DIR})
    find_file(OPENCL_LIB_DIR opencl/lib PATHS ${THIRD_PARTY_DIR})
    file(GLOB_RECURSE OPENCL_LIBS ${OPENCL_LIB_DIR}/*.lib)
  elseif(UNIX)
    file(GLOB OPENCL_LIBS /usr/lib/aarch64-linux-gnu/libOpenCL.*)
    if("${OPENCL_LIBS}" STREQUAL "")
      file(GLOB OPENCL_LIBS /usr/local/cuda/lib64/libOpenCL.*)
    endif()
    if("${OPENCL_LIBS}" STREQUAL "")
      file(GLOB OPENCL_LIBS /usr/lib/x86_64-linux-gnu/libOpenCL.*)
    endif()
    if("${OPENCL_LIBS}" STREQUAL "")
      file(GLOB OPENCL_LIBS /usr/rk3399-libs/lib64/libOpenCL.*)
    endif()
  elseif(APPLE)
    message(FATAL_ERROR "Not Implemented.")
  else()
    message(FATAL_ERROR "Unknown Platform")
  endif()
  message(STATUS "[OPENCL include] ${OPENCL_INCLUDE_DIR}")
  message(STATUS "[OPENCL libs] ${OPENCL_LIBS}")
  if("${OPENCL_LIBS}" STREQUAL "")
    message(STATUS "Could not find corrected library.")
  endif()
  include_directories(${OPENCL_INCLUDE_DIR})
  link_libraries(${OPENCL_LIBS})
  set(OPENCL_IMPORT "TRUE")
else()
  message(STATUS "[SYS] OPENCL has been imported.")
endif()