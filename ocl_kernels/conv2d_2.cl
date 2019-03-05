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

  if (ow >= out_w || oh >= out_h || oc >= out_c) return;

  int ih = oh * STRIDE_H - PAD_H;
  int iw = ow * STRIDE_W - PAD_W;
  int p_fc = oc * IN_C * FILTER_H * FILTER_W;
  int p_out = oc * out_h * out_w + oh * out_w + ow;

  float reg = 0;
  for (int ic = 0; ic < IN_C; ++ic) {
    int p_in = ic * in_h * in_w + ih * in_w + iw;
    int p_f = p_fc + ic * FILTER_H * FILTER_W;
    for (int fh = 0; fh < FILTER_H; ++fh) {
      for (int fw = 0; fw < FILTER_W; ++fw) {
        int ix = iw + fw * DILATION_W;
        int iy = ih + fh * DILATION_H;
        if (ix >= 0 && ix < in_w && iy >= 0 && iy < in_h) {
          reg += filter[p_f + fh * FILTER_W + fw] * in[p_in + fw * DILATION_W];
        }
      }
      p_in += in_w * DILATION_W;
    }
  }

  out[p_out] = reg + bias[oc];
}
