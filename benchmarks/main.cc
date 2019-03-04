
#include "ocl_helper.h"

void test_on_access() {}
void test_on_dtype() {}
void test_on_align() {}
void test_on_vector() {}
void test_on_memory() {}

void reduce_sum(cl_mem d, float* a, int n) {}

#define DTYPE float

int main() {
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
        h_in[pin] = 0.25;
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
                            d_filter,
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
      ocl_create_program(context, device, "ocl_kernels/convolution.cl");
  cl_kernel kernel = ocl_create_kernel(context, program, "conv2d");

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
      queue, kernel, 3, 0, global, local, NULL, NULL, NULL));

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

  double sum = 0;
  for (int i = 0; i < out_c * out_h * out_w; i++) sum += h_out[i];
  std::cout << sum << '\t' << sum / (out_c * out_h * out_w) << std::endl;
}