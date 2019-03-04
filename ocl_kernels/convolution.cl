#include "ocl_kernels/kernel_helper.h"

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

  int ih = oh * stride_h - pad_h;
  int iw = ow * stride_w - pad_w;
  int p_fc = oc * in_c * filter_h * filter_w;
  int p_out = oc * out_h * out_w + oh * out_w + ow;

  float reg = 0;
  for (int ic = 0; ic < in_c; ++ic) {
    int p_in = ic * in_h * in_w + ih * in_w + iw;
    int p_f = p_fc + ic * filter_h * filter_w;
    for (int fh = 0; fh < filter_h; ++fh) {
      for (int fw = 0; fw < filter_w; ++fw) {
        int ix = iw + fw * dilation_w;
        int iy = ih + fh * dilation_h;
        if (ix >= 0 && ix < in_w && iy >= 0 && iy < in_h) {
          reg += filter[p_f + fh * filter_w + fw] * in[p_in + fw * dilation_w];
        }
      }
      p_in += in_w * dilation_w;
    }
  }

  out[p_out] = reg + bias[oc];
}