/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_BINARY_COMMON_H
#define GPU_INTEL_BINARY_COMMON_H

#include "gpu/intel/include/dispatch.h"
#include "gpu/intel/include/dnnl_interop.h"
#include "gpu/intel/include/io.h"
#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"
#include "gpu/intel/include/utils.h"

#undef DST_OFF
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define SRC0_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC0, x0, x1, x2, x3, x4, x5)
#define SRC1_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC1, x0, x1, x2, x3, x4, x5)
#define SRC2_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC2, x0, x1, x2, x3, x4, x5)

#if NVECT == 1 || IS_PLAIN_LAYOUT
#define ELEM_DATA_T float
#elif NVECT == 2
#define ELEM_DATA_T float2
#elif NVECT == 4
#define ELEM_DATA_T float4
#elif NVECT == 8
#define ELEM_DATA_T float8
#endif

#define DEF_binary_op(dt, special_dt) \
    dt __attribute__((overloadable)) binary_op(int alg, dt src0, dt src1) { \
        switch (alg) { \
            case binary_add: return src0 + src1; \
            case binary_mul: return src0 * src1; \
            case binary_max: return max(src0, src1); \
            case binary_min: return min(src0, src1); \
            case binary_div: return src0 / src1; \
            case binary_sub: return src0 - src1; \
            case binary_ge: \
                return (src0 >= src1) ? SPECIAL(special_dt, one) \
                                      : SPECIAL(special_dt, zero); \
            case binary_gt: \
                return (src0 > src1) ? SPECIAL(special_dt, one) \
                                     : SPECIAL(special_dt, zero); \
            case binary_le: \
                return (src0 <= src1) ? SPECIAL(special_dt, one) \
                                      : SPECIAL(special_dt, zero); \
            case binary_lt: \
                return (src0 < src1) ? SPECIAL(special_dt, one) \
                                     : SPECIAL(special_dt, zero); \
            case binary_eq: \
                return (src0 == src1) ? SPECIAL(special_dt, one) \
                                      : SPECIAL(special_dt, zero); \
            case binary_ne: \
                return (src0 != src1) ? SPECIAL(special_dt, one) \
                                      : SPECIAL(special_dt, zero); \
        } \
        DEBUG_PRINT("Invalid binary op: %d\n", alg); \
        return SPECIAL(special_dt, max); \
    }

DEF_binary_op(float, float);
DEF_binary_op(float2, float);
DEF_binary_op(float4, float);
DEF_binary_op(float8, float);
#undef DEF_binary_op

#define DEF_ternary_op(dt, special_dt) \
    dt __attribute__((overloadable)) ternary_op( \
            int alg, dt src0, dt src1, char src2) { \
        return (src2 != 0) ? src0 : src1; \
    }

DEF_ternary_op(float, float);
DEF_ternary_op(float2, float);
DEF_ternary_op(float4, float);
DEF_ternary_op(float8, float);
#undef DEF_ternary_op

#endif
