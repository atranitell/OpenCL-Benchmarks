
#include "ocl_helper.h"

//------------------------------------------------------------------------------
// LOAD SOURCE AND APPEND DEFINES
//------------------------------------------------------------------------------
char* ocl_load_source_with_define(const char* filename,
                                  const char* preamble,
                                  size_t* szFinalLength) {
  // locals
  FILE* pFileStream = NULL;
  size_t szSourceLength;

  // open the OpenCL source code file
#ifdef _WIN32  // Windows version
  if (fopen_s(&pFileStream, filename, "rb") != 0) {
    return NULL;
  }
#else  // Linux version
  pFileStream = fopen(filename, "rb");
  if (pFileStream == 0) {
    return NULL;
  }
#endif

  size_t szPreambleLength = strlen(preamble);

  // get the length of the source code
  fseek(pFileStream, 0, SEEK_END);
  szSourceLength = ftell(pFileStream);
  fseek(pFileStream, 0, SEEK_SET);

  // allocate a buffer for the source code string and read it in
  char* cSourceString = (char*)malloc(szSourceLength + szPreambleLength + 1);
  memcpy(cSourceString, preamble, szPreambleLength);
  if (fread(
          (cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) !=
      1) {
    fclose(pFileStream);
    free(cSourceString);
    return 0;
  }

  // close the file and return the total length of the combined (preamble +
  // source) string
  fclose(pFileStream);
  if (szFinalLength != 0) {
    *szFinalLength = szSourceLength + szPreambleLength;
  }
  cSourceString[szSourceLength + szPreambleLength] = '\0';

  return cSourceString;
}

//------------------------------------------------------------------------------
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//------------------------------------------------------------------------------
cl_context ocl_create_context(size_t platformId) {
  cl_int err;
  cl_uint numPlatforms;
  cl_context context = NULL;
  cl_platform_id platforms[255];
  cl_platform_id selectedPlatform;

  // get all platform
  CL_CALL(clGetPlatformIDs(0, NULL, &numPlatforms));
  // select platform
  CL_CALL(clGetPlatformIDs(numPlatforms, platforms, NULL));
  selectedPlatform = platforms[platformId];
  // Next, create an OpenCL context on the platform.
  cl_context_properties contextProperties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)selectedPlatform, 0};
  context = clCreateContextFromType(
      contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
  CL_CALL(err);

  return context;
}

//------------------------------------------------------------------------------
//  Get device
//------------------------------------------------------------------------------
cl_device_id ocl_create_device(cl_context context, cl_device_id deviceId) {
  size_t BufferSize = -1;
  CL_CALL(clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &BufferSize));
  CL_CALL(clGetContextInfo(
      context, CL_CONTEXT_DEVICES, BufferSize, &deviceId, NULL));
  return deviceId;
}

//------------------------------------------------------------------------------
//  get command queue
//------------------------------------------------------------------------------
cl_command_queue ocl_create_command_queue(cl_context context,
                                          cl_device_id deviceId) {
  cl_int err;
  auto commandQueue_ =
      clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE, &err);
  CL_CALL(err);
  return commandQueue_;
}

//------------------------------------------------------------------------------
//  Create program
//------------------------------------------------------------------------------
cl_program ocl_create_program(cl_context context,
                              cl_device_id deviceId,
                              const std::string& filepath,
                              const std::vector<std::string>& defines,
                              const std::string& buildOption) {
  cl_int err;
  cl_program program;
  size_t program_length;
  std::ostringstream preamble;
  for (auto def : defines) preamble << def << '\n';

  // load source
  auto source = ocl_load_source_with_define(
      filepath.c_str(), preamble.str().c_str(), &program_length);

  // build from file
  program = clCreateProgramWithSource(
      context, 1, (const char**)&source, &program_length, &err);
  CL_CALL(err);

  // build program
  err = clBuildProgram(program, 1, &deviceId, buildOption.c_str(), NULL, NULL);
  if (err != CL_SUCCESS) {
    char buildLog[16384];
    clGetProgramBuildInfo(program,
                          deviceId,
                          CL_PROGRAM_BUILD_LOG,
                          sizeof(buildLog),
                          buildLog,
                          NULL);
    std::cerr << "When building " << filepath << ", " << ocl_status_code.at(err)
              << ", Error in program: " << buildLog;
  }

  return program;
}

//------------------------------------------------------------------------------
//  Create kernels
//------------------------------------------------------------------------------
cl_kernel ocl_create_kernel(cl_context context,
                            cl_program program,
                            const std::string& kernelName) {
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
  CL_CALL(err);
  return kernel;
}

//------------------------------------------------------------------------------
//  Next pow2
//------------------------------------------------------------------------------
size_t ocl_next_pow2(size_t x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}