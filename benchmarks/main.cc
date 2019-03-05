
#include "ocl_helper.h"

#define DTYPE float
#define TEXTURE

struct Timer {
  float queue;
  float submit;
  float run;
  float total;
};

void benchmark_on_buffer(const std::vector<std::string>& params) {
  cl_context context = ocl_create_context(0);
  cl_device_id deviceId;
  cl_device_id device = ocl_create_device(context, deviceId);
  cl_command_queue queue = ocl_create_command_queue(context, device);
  int err_code;

  // convolutional operators
  int in_c = 512;
  int in_h = 13;
  int in_w = 13;
  int out_c = 1024;
  int out_h = 13;
  int out_w = 13;
  int filter_h = 3;
  int filter_w = 3;
  int stride_h = 1;
  int stride_w = 1;
  int pad_h = 1;
  int pad_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;

  size_t global[3] = {16, 16, out_c};
  size_t local[3] = {16, 16, 1};

  DTYPE* h_in;
  DTYPE* h_out;
  DTYPE* h_filter;
  DTYPE* h_bias;

  cl_mem d_in;
  cl_mem d_out;
  cl_mem d_filter;
  cl_mem d_bias;

  // allocate memory on host
  h_in = (DTYPE*)malloc(in_c * in_h * in_w * sizeof(DTYPE));
  h_out = (DTYPE*)malloc(out_c * out_h * out_w * sizeof(DTYPE));
  h_filter = (DTYPE*)malloc(out_c * in_c * filter_h * filter_w * sizeof(DTYPE));
  h_bias = (DTYPE*)malloc(out_c * sizeof(DTYPE));

  // in
  for (int c = 0; c < in_c; c++) {
    for (int h = 0; h < in_h; h++) {
      for (int w = 0; w < in_w; w++) {
        int pin = c * in_h * in_w + h * in_w + w;
        h_in[pin] = 0.01*c;
      }
    }
  }

  // filter
  for (int oc = 0; oc < out_c; oc++) {
    for (int ic = 0; ic < in_c; ic++) {
      for (int h = 0; h < filter_h; h++) {
        for (int w = 0; w < filter_w; w++) {
          int pin = oc * in_c * filter_h * filter_w + ic * filter_h * filter_w +
                    h * filter_w + w;
          h_filter[pin] = 1.5;
        }
      }
    }
  }

  // bias
  for (int oc = 0; oc < out_c; oc++) {
    h_bias[oc] = 0.5;
  }

  // allocate memory on device
  d_in = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        in_c * in_h * in_w * sizeof(DTYPE),
                        NULL,
                        &err_code);
  CL_CALL(err_code);
  CL_CALL(clEnqueueWriteBuffer(queue,
                               d_in,
                               CL_TRUE,
                               0,
                               in_c * in_h * in_w * sizeof(DTYPE),
                               h_in,
                               0,
                               NULL,
                               NULL));

  d_out = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         out_c * out_h * out_w * sizeof(DTYPE),
                         NULL,
                         &err_code);
  CL_CALL(err_code);

  d_filter = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            out_c * in_c * filter_h * filter_w * sizeof(DTYPE),
                            NULL,
                            &err_code);
  CL_CALL(err_code);
  CL_CALL(
      clEnqueueWriteBuffer(queue,
                           d_filter,
                           CL_TRUE,
                           0,
                           out_c * in_c * filter_h * filter_w * sizeof(DTYPE),
                           h_filter,
                           0,
                           NULL,
                           NULL));

  d_bias = clCreateBuffer(
      context, CL_MEM_READ_WRITE, out_c * sizeof(DTYPE), NULL, &err_code);
  CL_CALL(err_code);
  CL_CALL(clEnqueueWriteBuffer(
      queue, d_bias, CL_TRUE, 0, out_c * sizeof(DTYPE), h_bias, 0, NULL, NULL));
  clFinish(queue);

  cl_program program =
      ocl_create_program(context, device, std::string("ocl_kernels/") + params[1].c_str());
  cl_kernel kernel = ocl_create_kernel(context, program, params[2].c_str());
  cl_event event;

  // execute
  for (int i = 0; i < 5; i++) {
    // set args
    int n = 0;
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_mem), &d_in));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_mem), &d_out));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_mem), &d_filter));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_mem), &d_bias));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &in_c));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &in_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &in_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &out_c));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &out_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &out_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &filter_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &filter_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &stride_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &stride_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &pad_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &pad_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &dilation_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &dilation_w));
    CL_CALL(clEnqueueNDRangeKernel(
        queue, kernel, 3, 0, global, local, NULL, NULL, &event));
    clWaitForEvents(1, &event);

    clFinish(queue);
    CL_CALL(clEnqueueReadBuffer(queue,
                                d_out,
                                CL_TRUE,
                                0,
                                out_c * out_h * out_w * sizeof(DTYPE),
                                h_out,
                                0,
                                NULL,
                                NULL));

    cl_ulong time_submit;
    cl_ulong time_queue;
    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_SUBMIT,
                            sizeof(time_submit),
                            &time_submit,
                            NULL);
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_QUEUED,
                            sizeof(time_queue),
                            &time_queue,
                            NULL);
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_START,
                            sizeof(time_start),
                            &time_start,
                            NULL);
    clGetEventProfilingInfo(
        event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    Timer timer;
    timer.queue = (time_submit - time_queue) / 1000.0 / 1000.0;
    timer.submit = (time_start - time_queue) / 1000.0 / 1000.0;
    timer.run = (time_end - time_start) / 1000.0 / 1000.0;
    timer.total = (time_end - time_submit) / 1000.0 / 1000.0;
    printf("Idx: %2d, Queue: %.3f, Submit: %.3f, Run: %.3f, Total: %.3f\n",
           i,
           timer.queue,
           timer.submit,
           timer.run,
           timer.total);
  }

  double sum = 0;
  for (int i = 0; i < out_c * out_h * out_w; i++) sum += h_out[i];
  std::cout << sum << '\t' << sum / (out_c * out_h * out_w) << std::endl;

  free(h_in);
  free(h_out);
  free(h_filter);
  free(h_bias);

  clReleaseMemObject(d_in);
  clReleaseMemObject(d_out);
  clReleaseMemObject(d_filter);
  clReleaseMemObject(d_bias);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void benchmark_on_image(const std::vector<std::string>& params) {
  cl_context context = ocl_create_context(0);
  cl_device_id deviceId;
  cl_device_id device = ocl_create_device(context, deviceId);
  cl_command_queue queue = ocl_create_command_queue(context, device);
  int err_code;

  // convolutional operators
  int in_c = 512;
  int in_h = 13;
  int in_w = 13;
  int out_c = 1024;
  int out_h = 13;
  int out_w = 13;
  int filter_h = 3;
  int filter_w = 3;
  int stride_h = 1;
  int stride_w = 1;
  int pad_h = 1;
  int pad_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;

  size_t global[3] = {16, 16, out_c};
  size_t local[3] = {16, 16, 1};

  DTYPE* h_in;
  DTYPE* h_out;
  DTYPE* h_filter;
  DTYPE* h_bias;

  cl_mem d_in;
  cl_mem d_out;
  cl_mem d_filter;
  cl_mem d_bias;

  // allocate memory on host
  h_in = (DTYPE*)malloc(in_c * in_h * in_w * sizeof(DTYPE));
  h_out = (DTYPE*)malloc(out_c * out_h * out_w * sizeof(DTYPE));
  h_filter = (DTYPE*)malloc(out_c * in_c * filter_h * filter_w * sizeof(DTYPE));
  h_bias = (DTYPE*)malloc(out_c * sizeof(DTYPE));

  // in
  for (int c = 0; c < in_c; c++) {
    for (int h = 0; h < in_h; h++) {
      for (int w = 0; w < in_w; w++) {
        int pin = c * in_h * in_w + h * in_w + w;
        h_in[pin] = 0.01*c;
      }
    }
  }

  // filter
  for (int oc = 0; oc < out_c; oc++) {
    for (int ic = 0; ic < in_c; ic++) {
      for (int h = 0; h < filter_h; h++) {
        for (int w = 0; w < filter_w; w++) {
          int pin = oc * in_c * filter_h * filter_w + ic * filter_h * filter_w +
                    h * filter_w + w;
          h_filter[pin] = 1.5;
        }
      }
    }
  }

  // bias
  for (int oc = 0; oc < out_c; oc++) {
    h_bias[oc] = 0.5;
  }

  // allocate memory on device
  d_in = clCreateBuffer(context,
                        CL_MEM_READ_WRITE,
                        in_c * in_h * in_w * sizeof(DTYPE),
                        NULL,
                        &err_code);
  CL_CALL(err_code);
  CL_CALL(clEnqueueWriteBuffer(queue,
                               d_in,
                               CL_TRUE,
                               0,
                               in_c * in_h * in_w * sizeof(DTYPE),
                               h_in,
                               0,
                               NULL,
                               NULL));

  cl_image_format fmt;
  fmt.image_channel_data_type = CL_FLOAT;
  fmt.image_channel_order = CL_INTENSITY;
  auto t_in = clCreateImage3D(context,
                              CL_MEM_READ_WRITE,
                              &fmt,
                              in_w,
                              in_h,
                              in_c,
                              0,
                              0,
                              NULL,
                              &err_code);
  CL_CALL(err_code);
  size_t region[3] = {in_w, in_h, in_c};
  size_t offset[3] = {0, 0, 0};
  CL_CALL(clEnqueueCopyBufferToImage(queue,
                                     d_in,
                                     t_in,
                                     0,
                                     offset,
                                     region,
                                     0,
                                     NULL,
                                     NULL));

  CL_CALL(clEnqueueReadImage(queue,
                             t_in,
                             CL_TRUE,
                             offset,
                             region,
                             in_w * sizeof(DTYPE),
                             in_h * in_w * sizeof(DTYPE),
                             h_in,
                             0,
                             NULL,
                             NULL));
//  double sum_in = 0;
//  for (int i = 0; i < 512*13*13; i++) {
//    sum_in += h_in[i];
//  }
//  printf("%.5f\n", sum_in);

  d_out = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         out_c * out_h * out_w * sizeof(DTYPE),
                         NULL,
                         &err_code);
  CL_CALL(err_code);

  d_filter = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            out_c * in_c * filter_h * filter_w * sizeof(DTYPE),
                            NULL,
                            &err_code);
  CL_CALL(err_code);
  CL_CALL(
      clEnqueueWriteBuffer(queue,
                           d_filter,
                           CL_TRUE,
                           0,
                           out_c * in_c * filter_h * filter_w * sizeof(DTYPE),
                           h_filter,
                           0,
                           NULL,
                           NULL));

  d_bias = clCreateBuffer(
      context, CL_MEM_READ_WRITE, out_c * sizeof(DTYPE), NULL, &err_code);
  CL_CALL(err_code);
  CL_CALL(clEnqueueWriteBuffer(
      queue, d_bias, CL_TRUE, 0, out_c * sizeof(DTYPE), h_bias, 0, NULL, NULL));
  clFinish(queue);

  cl_program program =
      ocl_create_program(context, device, std::string("ocl_kernels/") + params[1].c_str());
  cl_kernel kernel = ocl_create_kernel(context, program, params[2].c_str());
  cl_event event;

  // execute
  for (int i = 0; i < 5; i++) {
    // set args
    int n = 0;
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_mem), &t_in));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_mem), &d_out));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_mem), &d_filter));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_mem), &d_bias));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &in_c));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &in_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &in_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &out_c));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &out_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &out_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &filter_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &filter_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &stride_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &stride_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &pad_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &pad_w));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &dilation_h));
    CL_CALL(clSetKernelArg(kernel, n++, sizeof(cl_int), &dilation_w));
    CL_CALL(clEnqueueNDRangeKernel(
        queue, kernel, 3, 0, global, local, NULL, NULL, &event));
    clWaitForEvents(1, &event);

    clFinish(queue);
    CL_CALL(clEnqueueReadBuffer(queue,
                                d_out,
                                CL_TRUE,
                                0,
                                out_c * out_h * out_w * sizeof(DTYPE),
                                h_out,
                                0,
                                NULL,
                                NULL));

    cl_ulong time_submit;
    cl_ulong time_queue;
    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_SUBMIT,
                            sizeof(time_submit),
                            &time_submit,
                            NULL);
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_QUEUED,
                            sizeof(time_queue),
                            &time_queue,
                            NULL);
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_START,
                            sizeof(time_start),
                            &time_start,
                            NULL);
    clGetEventProfilingInfo(
        event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    Timer timer;
    timer.queue = (time_submit - time_queue) / 1000.0 / 1000.0;
    timer.submit = (time_start - time_queue) / 1000.0 / 1000.0;
    timer.run = (time_end - time_start) / 1000.0 / 1000.0;
    timer.total = (time_end - time_submit) / 1000.0 / 1000.0;
    printf("Idx: %2d, Queue: %.3f, Submit: %.3f, Run: %.3f, Total: %.3f\n",
           i,
           timer.queue,
           timer.submit,
           timer.run,
           timer.total);
  }

  double sum = 0;
//  for (int i = 0; i < 13; i++)
//    for (int j = 0; j < 13; j++)
//      printf("%d %d %f\n", i, j, *(h_out + 13*i + j));
  for (int i = 0; i < out_c * out_h * out_w; i++) sum += h_out[i];
  std::cout << sum << '\t' << sum / (out_c * out_h * out_w) << std::endl;

  free(h_in);
  free(h_out);
  free(h_filter);
  free(h_bias);

  clReleaseMemObject(d_in);
  clReleaseMemObject(t_in);
  clReleaseMemObject(d_out);
  clReleaseMemObject(d_filter);
  clReleaseMemObject(d_bias);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void benchmark_all(std::vector<std::string>& params) {
  printf("---- Start to benchmark all conv implementation ----\n");
  params[2] = "conv2d";

  printf("[CONV: Traditional Method]\n");
  params[1] = "conv2d_1.cl";
  benchmark_on_buffer(params);

  printf("[CONV: Using predefined Macors]\n");
  params[1] = "conv2d_2.cl";
  benchmark_on_buffer(params);

  printf("[CONV: Unrolling Loop with diminished operation]\n");
  params[1] = "conv2d_3.cl";
  benchmark_on_buffer(params);

  printf("[CONV: Unrolling Loop]\n");
  params[1] = "conv2d_3_1.cl";
  benchmark_on_buffer(params);

  printf("[CONV: Internal Math function]\n");
  params[1] = "conv2d_4.cl";
  benchmark_on_buffer(params);

  printf("[CONV: Divide processing border]\n");
  params[1] = "conv2d_5.cl";
  benchmark_on_buffer(params);

  printf("[CONV: Vector load]\n");
  params[1] = "conv2d_6.cl";
  benchmark_on_buffer(params);

  printf("[CONV: Divide border and unrolling]\n");
  params[1] = "conv2d_7.cl";
  benchmark_on_buffer(params);

  printf("[CONV: Best on buffer]\n");
  params[1] = "conv2d_8.cl";
  benchmark_on_buffer(params);

  // image with buffer
  printf("[CONV: Image Sampler]\n");
  params[1] = "conv2d_image.cl";
  benchmark_on_image(params);
}

int main(int argc, char **argv) {
  // convert argv to vector<string>
  std::vector<std::string> params;
  params.resize(10);

  for (int i = 0; i < argc; i++)
    params[i] = std::string(argv[i]);

  if (argc == 1) {
    benchmark_all(params);
  } 
  else {
#ifdef TEXTURE
    benchmark_on_image(params);
#else
    benchmark_on_buffer(params);
#endif
  }

  return 0;
}