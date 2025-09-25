/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#define DT_UNDEF 1
#include "gpu/intel/include/types.h"
#include "gpu/intel/include/types_interop.h"

__kernel void mx_scale_dst(__global float *restrict src,
        __global uchar *restrict dst, __global uchar *restrict dst_scales,
        int groupSize, long D0, long D1, long D2, long c_stride_d3,
        long c_stride_d2, long c_stride_d1, long c_stride_d0, long c_stride_m,
        long c_stride_n) {
    long m = get_global_id(0);
    long n = get_global_id(1);
    int mb = get_global_id(2);
    // decompose mb into batch dimensions (d0..d3)
    long d3 = mb / D0 / D1 / D2;
    long d2 = (mb / D0 / D1) % D2;
    long d1 = (mb / D0) % D1;
    long d0 = mb % D0;
    float max_group = 0;

    for (int i = 0; i < groupSize; ++i) {
        long off = 0;
        long n_iter = n * groupSize + i;
#if RUNTIME_DIMS
        off = offset6D(m, n_iter, d0, d1, d2, d3, c_stride_m, c_stride_n,
                c_stride_d0, c_stride_d1, c_stride_d2, c_stride_d3);
#else
#if NDIMS == 5
        off = DST_OFF(d2 % DST_D0, d1 % DST_D1, d0 % DST_D2, m, n_iter);
#elif NDIMS == 4
        off = DST_OFF(d1 % DST_D0, d0 % DST_D1, 0, m, n_iter);
#elif NDIMS == 3
        off = DST_OFF(d0 % DST_D0, m, 0, 0, n_iter);
#else
        off = DST_OFF(m, n_iter, 0, 0, 0);
#endif
#endif
        max_group = max(max_group, fabs(src[off]));
    }

#define E8M0(x) cvt_e8m0_to_f32(cvt_f32_to_e8m0(x))
    float scale_val = max(E8M0(0.f), E8M0(max_group) / E8M0(DST_DATA_FMAX));
#undef E8M0

    for (int i = 0; i < groupSize; ++i) {
        long off = 0;
        long n_iter = n * groupSize + i;
#if RUNTIME_DIMS
        off = offset6D(m, n_iter, d0, d1, d2, d3, c_stride_m, c_stride_n,
                c_stride_d0, c_stride_d1, c_stride_d2, c_stride_d3);
#else
#if NDIMS == 5
        off = DST_OFF(d2 % DST_D0, d1 % DST_D1, d0 % DST_D2, m, n_iter);
#elif NDIMS == 4
        off = DST_OFF(d1 % DST_D0, d0 % DST_D1, 0, m, n_iter);
#elif NDIMS == 3
        off = DST_OFF(d0 % DST_D0, m, 0, 0, n_iter);
#else
        off = DST_OFF(m, n_iter, 0, 0, 0);
#endif
#endif
        dst[off] = TO_DST(src[off] / scale_val);
    }

    long scale_off = 0;
#if RUNTIME_DIMS
    scale_off = offset6D(m, n, d0, d1, d2, d3, c_stride_m / 1,
            c_stride_n / groupSize, c_stride_d0, c_stride_d1, c_stride_d2,
            c_stride_d3);
#else
#if NDIMS == 5
    scale_off = DST_SCALE_OFF(
            d2 % DST_D0, d1 % DST_D1, d0 % DST_D2, m, n, groupSize, 1);
#elif NDIMS == 4
    scale_off = DST_SCALE_OFF(d1 % DST_D0, d0 % DST_D1, 0, m, n, groupSize, 1);
#elif NDIMS == 3
    scale_off = DST_SCALE_OFF(d0 % DST_D0, m, 0, 0, n, groupSize, 1);
#else
    scale_off = DST_SCALE_OFF(m, n, 0, 0, 0, 1, groupSize);
#endif
#endif
    dst_scales[scale_off] = cvt_f32_to_e8m0(scale_val);
}
