#ifndef _OCL_HELPER_H_
#define _OCL_HELPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "ocl_status.h"

#define CL_CALL(x)                                                  \
  {                                                                 \
    cl_int status = (x);                                            \
    if (status != CL_SUCCESS)                                       \
      std::cerr << "OpenCL calling error, code: " << status << ", " \
                << ocl_status_code.at(status);                      \
    std::cout << __FUNCTION__ << " " << __LINE__ << " " << status   \
              << std::endl;                                         \
  }

//------------------------------------------------------------------------------
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//------------------------------------------------------------------------------
cl_context ocl_create_context(size_t platformId);

//------------------------------------------------------------------------------
//  Get device
//------------------------------------------------------------------------------
cl_device_id ocl_create_device(cl_context context, cl_device_id device_id);

//------------------------------------------------------------------------------
//  get command queue
//------------------------------------------------------------------------------
cl_command_queue ocl_create_command_queue(cl_context context,
                                          cl_device_id deviceId);

//------------------------------------------------------------------------------
//  Create program
//------------------------------------------------------------------------------
cl_program ocl_create_program(cl_context context,
                              cl_device_id deviceId,
                              const std::string& filepath,
                              const std::vector<std::string>& defines = {},
                              const std::string& buildOption = "");
//------------------------------------------------------------------------------
//  Create kernels
//------------------------------------------------------------------------------
cl_kernel ocl_create_kernel(cl_context context,
                            cl_program program,
                            const std::string& kernelName);

//------------------------------------------------------------------------------
//  Next pow2
//------------------------------------------------------------------------------
size_t ocl_next_pow2(size_t x);

#endif