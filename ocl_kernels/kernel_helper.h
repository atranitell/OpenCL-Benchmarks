// MIT License

// Copyright (c) 2018 kaiJIN

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _KERNEL_HELPER_OPENCL_H_
#define _KERNEL_HELPER_OPENCL_H_

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// group id
#define GROUP_ID                                             \
  (get_num_groups(0) * get_num_groups(1) * get_group_id(2) + \
   get_num_groups(0) * get_group_id(1) + get_group_id(0))

// local id
#define LOCAL_ID                                             \
  (get_local_size(0) * get_local_size(1) * get_local_id(2) + \
   get_local_size(0) * get_local_id(1) + get_local_id(0))

// local size
#define LOCAL_SIZE (get_local_size(0) * get_local_size(1) * get_local_size(2))

// global id
#define GLOBAL_ID (GROUP_ID * LOCAL_SIZE + LOCAL_ID)

#endif