#include "ocl_kernels/kernel_helper.h"

#define FILTER_H 3
#define FILTER_W 3
#define DILATION_H 1
#define DILATION_W 1
#define PAD_H 1
#define PAD_W 1
#define STRIDE_H 1
#define STRIDE_W 1
#define IN_C 512
#define OUT_C 1024

__kernel void conv2d(__global const float* in,
                     __global float* out,
                     __global float* filter,
                     __global float* bias,
                     const int in_c,
                     const int in_h,
                     const int in_w,
                     const int out_c,
                     const int out_h,
                     const int out_w,
                     const int filter_h,
                     const int filter_w,
                     const int stride_h,
                     const int stride_w,
                     const int pad_h,
                     const int pad_w,
                     const int dilation_h,
                     const int dilation_w) {
  int ow = get_global_id(0);
  int oh = get_global_id(1);
  int oc = get_global_id(2);

  if (ow >= out_w || oh >= out_h || oc >= OUT_C) return;

  int ih = oh * STRIDE_H - PAD_H;
  int iw = ow * STRIDE_W - PAD_W;
  int p_fc = oc * IN_C * FILTER_H * FILTER_W;
  int p_out = oc * out_h * out_w + oh * out_w + ow;

  float reg = 0;
  int ix = 0;
  int iy = 0;
  for (int ic = 0; ic < IN_C; ++ic) {
    int p_in = ic * in_h * in_w + ih * in_w + iw;
    int p_f = p_fc + ic * FILTER_H * FILTER_W;
    
    if (iw >= 0 && iw < in_w && ih >= 0 && ih < in_h) {
      reg += filter[p_f] * in[p_in];
    }

    if (iw + DILATION_W >= 0 && iw + DILATION_W < in_w && ih >= 0 && ih < in_h) {
      reg += filter[p_f + 1] * in[p_in + DILATION_W];
    }

    if (iw + 2 * DILATION_W >= 0 && iw + 2 * DILATION_W < in_w && ih >= 0 && ih < in_h) {
      reg += filter[p_f + 2] * in[p_in + 2 * DILATION_W];
    }

    p_in += in_w * DILATION_W;

    if (iw >= 0 && iw < in_w && ih + DILATION_H >= 0 && ih + DILATION_H < in_h) {
      reg += filter[p_f + FILTER_W + 0] * in[p_in];
    }

    if (iw + DILATION_W >= 0 && iw + DILATION_W < in_w && ih + DILATION_H >= 0 && ih + DILATION_H < in_h) {
      reg += filter[p_f + FILTER_W + 1] * in[p_in + DILATION_W];
    }

    if (iw + 2 * DILATION_W >= 0 && iw + 2 * DILATION_W < in_w && ih + DILATION_H >= 0 && ih + DILATION_H < in_h) {
      reg += filter[p_f + FILTER_W + 2] * in[p_in + 2 * DILATION_W];
    }

    p_in += in_w * DILATION_W;

    if (iw >= 0 && iw < in_w && ih + 2 * DILATION_H >= 0 && ih + 2 * DILATION_H < in_h) {
      reg += filter[p_f + 2 * FILTER_W + 0] * in[p_in];
    }

    if (iw + DILATION_W >= 0 && iw + DILATION_W < in_w && ih + 2 * DILATION_H >= 0 && ih + 2 * DILATION_H < in_h) {
      reg += filter[p_f + 2 * FILTER_W + 1] * in[p_in + DILATION_W];
    }

    if (iw + 2 * DILATION_W >= 0 && iw + 2 * DILATION_W < in_w && ih + 2 * DILATION_H >= 0 && ih + 2 * DILATION_H < in_h) {
      reg += filter[p_f + 2 * FILTER_W + 2] * in[p_in + 2 * DILATION_W];
    }
  }

  out[p_out] = reg + bias[oc];
}
