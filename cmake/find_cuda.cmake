# Find CUDA libraries
# Author: Kai Jin
# Updated: 09/18/2018
# mail: <atranitell@gmail.com>

# CUDA_FOUND
# CUDA_INCLUDE_DIR
# CUDA_LIBRARIES

if(NOT CUDA_IMPORT STREQUAL "TRUE")
  if(WIN32 OR UNIX)
    find_package(CUDA REQUIRED)
  elseif(UNIX)
    message(FATAL_ERROR "Not Implemented.")
  elseif(APPLE)
    message(FATAL_ERROR "Not Implemented.")
  else()
    message(FATAL_ERROR "Unknown Platform")
  endif()
  # configure cuda
  set(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
  if(WIN32)
    set(CUDA_LIBS
      ${CUDA_LIBRARIES}
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudnn.lib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cublas.lib)
  else()
    file(GLOB_RECURSE CUDA_LIBS 
      /usr/local/cuda/lib64/libcublas.so
      /usr/local/cuda/lib64/libcudnn.so
      /usr/local/cuda/lib64/libcudart.so
      /usr/local/cuda/lib64/libcurand.so)
  endif()
  message(STATUS "[CUDA include] ${CUDA_INCLUDE_DIR}")
  message(STATUS "[CUDA libs] ${CUDA_LIBS}")
  include_directories(${CUDA_INCLUDE_DIR})
  link_libraries(${CUDA_LIBS})
  set(CUDA_IMPORT "TRUE")
else()
  message(STATUS "[SYS] CUDA has been imported.")
endif()