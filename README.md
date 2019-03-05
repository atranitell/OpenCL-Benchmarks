
#### Device Attributes
- Device: Mali T860
- OpenCL Version: v1.2
- Max clock frequency : 200MHz
- Max compute units: 4

#### Convolution Ops Analysis
- inputs: [1, 512, 13, 13]
- weights: [1024, 512, 3, 3], stride 1, padding 1
- bias: [1024]
- outputs: [1, 1024, 13, 13]

#### Setting: normal
```c++
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
```
| Setting | Queue  | Submit | Run     | Total    |
|:-------:|:------:|:------:|:-------:|:--------:|
| normal  | 0.146  | 1.193  | 913.301 | 913.750  |


#### Setting: predefined macros
we could pass the macros to unrolling the convolution ops.
- s1: filter
```
#define FILTER_H 3
#define FILTER_W 3
```
- s2: filter + dilation
```
#define DILATION_H 1
#define DILATION_W 1
```
- s3: filter + dilation + pad + stride
```
#define PAD_H 1
#define PAD_W 1
#define STRIDE_H 1
#define STRIDE_W 1
```
- s4: filter + dilation + pad + stride + in
```
#define IN_C 512
```
| Setting   | Queue  | Submit | Run     | Total    |
|:---------:|:------:|:------:|:-------:|:--------:|
| macros s1 | 0.166  | 0.579  | 961.758 | 962.171  |
| macros s2 | 0.100  | 0.510  | 747.560 | 747.969  |
| macros s3 | 0.131  | 0.862  | 747.498 | 748.229  |
| macros s4 | 0.043  | 0.357  | 746.977 | 747.291  |


#### Setting: unrolling
the for statement could be unrolled by a sequence structure.
- s1: unrolling and reduce unnessary ops
```
iw + 0 * DILATION_W --> iw
iw + 1 * DILATION_W --> iw + DILATION_W
```
- s2: keep structure
```
iw + 0 * DILATION_W + 0
iw + 1 * DILATION_W + 1
```
| Setting   | Queue  | Submit | Run     | Total    |
|:---------:|:------:|:------:|:-------:|:--------:|
| unroll s1 | 0.151  | 0.571  | 817.677 | 818.097  |
| unroll s2 | 0.104  | 0.821  | 793.995 | 794.712  |

#### Setting: Internal function
using internal function instead of normal ops
```
a + b * c -> mad24(b, c, a); / fma(b, c, a);
```
| Setting   | Queue  | Submit | Run     | Total    |
|:---------:|:------:|:------:|:-------:|:--------:|
| innner s1 | 0.095  | 0.758  | 747.630 | 248.293  |


#### Setting: optimize operator
divided into 2 parts: one without condition statements, another (border) with condition.
```

```
| Setting   | Queue  | Submit | Run     | Total    |
|:---------:|:------:|:------:|:-------:|:--------:|
| conv s1   | 0.072  | 0.884  | 579.417 | 580.229  |

- border -> 197 ms
- non-border -> 270 ms
