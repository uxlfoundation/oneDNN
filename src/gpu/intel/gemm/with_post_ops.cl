/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include "gpu/intel/include/dispatch.h"
#include "gpu/intel/include/io.h"
#include "gpu/intel/include/math_utils.h"
#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"

#undef SRC_OFF
#define SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define BIAS_OFF(x0, x1, x2, x3, x4, x5) \
    OFF_MD(BIAS, (x0 % BIAS_PD0), (x1 % BIAS_PD1), (x2 % BIAS_PD2), \
            (x3 % BIAS_PD3), (x4 % BIAS_PD4), (x5 % BIAS_PD5))

__kernel void gemm_post_ops(__global SRC_DATA_T *src,
        __global BIAS_DATA_T *bias, __global DST_DATA_T *dst POST_OP_ARGS,
#if WITH_HOST_SRC_SCALE
        float a_scale_value,
#else
        global float *a_scales,
#endif
#if WITH_HOST_WEI_SCALE
        WEI_SCALES_DATA_T b_scale_value,
#else
        global WEI_SCALES_DATA_T *b_scales,
#endif
#if WITH_HOST_DST_SCALE
        DST_SCALES_DATA_T c_scale_value,
#else
        global DST_SCALES_DATA_T *c_scales,
#endif
        int scale_stride,
#if WITH_HOST_DST_ZP
        int dst_zp_value
#else
        global int *dst_zp
#endif
) {
#if WITH_HOST_SRC_SCALE
    float *a_scales = &a_scale_value;
#endif
#if WITH_HOST_WEI_SCALE
    WEI_SCALES_DATA_T *b_scales = &b_scale_value;
#endif
#if WITH_HOST_DST_SCALE
    DST_SCALES_DATA_T *c_scales = &c_scale_value;
#endif
#if WITH_HOST_DST_ZP
    int *dst_zp = &dst_zp_value;
#endif

    const uint d0 = GWS_GET_D0();
    const uint d1 = GWS_GET_D1();
    const uint d2 = GWS_GET_D2();
    const uint d3 = GWS_GET_D3();

    size_t data_idx = SRC_OFF(d0, d1, d2, d3, 0, 0);

    ACC_DATA_T acc = load(acc, src, data_idx);
    POST_OP_DATA_T accumulator = 0;
    if (d0 < DST_D0 && d1 < DST_D1 && d2 < DST_D2 && d3 < DST_D3) {
        const float a_scale = A_SCALES ? a_scales[0] : 1;
        const uint b_scale_dim = (NDIMS == 2) ? d1 : (NDIMS == 3) ? d2 : d3;
        float b_scale = 1;
        if (B_SCALES) load(&b_scale, b_scales + scale_stride * b_scale_dim);
        if (A_SCALES || B_SCALES) acc *= a_scale * b_scale;
        if (bias) {
            ACC_DATA_T b = load(b, bias + BIAS_OFF(d0, d1, d2, d3, 0, 0));
            acc += b;
        }

        // Apply postops
        POST_OP_DATA_T sum_src = WITH_SUM ? load(sum_src, dst, data_idx) : 0.0f;

        accumulator = AS_POST_OP_DATA_T(acc);
        APPLY_POST_OPS_SERIAL(accumulator, sum_src, d0, d1, d2, d3, 0, 0);
#if WITH_DYNAMIC_DST_SCALE == 0
        if (C_SCALES) {
            POST_OP_DATA_T c_scale = load(c_scale, c_scales);
            accumulator /= c_scale;
        }
#endif
        if (DST_ZERO_POINT) accumulator += dst_zp[0];
    }
    write(dst + data_idx, accumulator);
}
