
#include "ocl_kernels/kernel_helper.h"

__kernel void element_add(__global float* a, 
                          __global float* b, 
                          __global float* c, 
                          int n) {
  unsigned int i = GLOBAL_ID;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}