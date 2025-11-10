/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_INTEL_INCLUDE_POST_OPS_H
#define GPU_INTEL_INCLUDE_POST_OPS_H

#ifndef POST_OP_DATA_T
#if DT_F64
#define POST_OP_DATA_T double
#else
#define POST_OP_DATA_T float
#endif
#endif

#if WITH_POST_OP
#include "gpu/intel/include/eltwise.h"
#include "gpu/intel/include/io.h"

float fwd_binary(unsigned algorithm, POST_OP_DATA_T x, POST_OP_DATA_T y) {
    switch (algorithm) {
        // binary
        case BINARY_ADD: return x + y; break;
        case BINARY_MUL: return x * y; break;
        case BINARY_MIN: return x < y ? x : y; break;
        case BINARY_MAX: return x > y ? x : y; break;
        case BINARY_DIV: return x / y; break;
        case BINARY_SUB: return x - y; break;
        case BINARY_GE: return x >= y; break;
        case BINARY_GT: return x > y; break;
        case BINARY_LE: return x <= y; break;
        case BINARY_LT: return x < y; break;
        case BINARY_EQ: return x == y; break;
        case BINARY_NE: return x != y; break;
        case RELU: // binary && relu = prelu
            return fwd_eltwise_common(RELU, x, y, 0.0f, 1.0f);
            break;
        default: return 0.f;
    }
}

// unused arguments are maintained for interface compatibility
#define APPLY_PO_BINARY(idx, acc, _sum_src, x0, x1, x2, x3, x4, x5) \
    { \
        const auto po_off \
                = OFF_RMD(CONCAT2(PO_, idx), x0, x1, x2, x3, x4, x5); \
        POST_OP_DATA_T po_src \
                = load(po_src, (CONCAT3(po, idx, _binary_arg)) + po_off); \
        acc = fwd_binary(CONCAT3(PO_, idx, _ALG), acc, po_src); \
    }

// unused arguments are maintained for interface compatibility
#define APPLY_PO_SUM(idx, acc, sum_src, _x0, _x1, _x2, _x3, _x4, _x5) \
    acc += (load(acc, &sum_src) - (CONCAT3(po, idx, _zp))) \
            * CONCAT3(po, idx, _scale);

// unused arguments are maintained for interface compatibility
#define APPLY_PO_ELTWISE(idx, acc, _sum_src, _x0, _x1, _x2, _x3, _x4, _x5) \
    acc = fwd_eltwise_common(CONCAT3(PO_, idx, _ALG), acc, \
            CONCAT3(po, idx, _alpha), CONCAT3(po, idx, _beta), \
            CONCAT3(po, idx, _scale));

// clang-format off
#define APPLY_PO_STAGE_0(...)
#define APPLY_PO_STAGE_1(...) APPLY_PO_0(0, __VA_ARGS__)
#define APPLY_PO_STAGE_2(...) APPLY_PO_STAGE_1(__VA_ARGS__) APPLY_PO_1(1, __VA_ARGS__)
#define APPLY_PO_STAGE_3(...) APPLY_PO_STAGE_2(__VA_ARGS__) APPLY_PO_2(2, __VA_ARGS__)
#define APPLY_PO_STAGE_4(...) APPLY_PO_STAGE_3(__VA_ARGS__) APPLY_PO_3(3, __VA_ARGS__)
#define APPLY_PO_STAGE_5(...) APPLY_PO_STAGE_4(__VA_ARGS__) APPLY_PO_4(4, __VA_ARGS__)
#define APPLY_PO_STAGE_6(...) APPLY_PO_STAGE_5(__VA_ARGS__) APPLY_PO_5(5, __VA_ARGS__)
#define APPLY_PO_STAGE_7(...) APPLY_PO_STAGE_6(__VA_ARGS__) APPLY_PO_6(6, __VA_ARGS__)
#define APPLY_PO_STAGE_8(...) APPLY_PO_STAGE_7(__VA_ARGS__) APPLY_PO_7(7, __VA_ARGS__)
#define APPLY_PO_STAGE_9(...) APPLY_PO_STAGE_8(__VA_ARGS__) APPLY_PO_8(8, __VA_ARGS__)
#define APPLY_PO_STAGE_10(...) APPLY_PO_STAGE_9(__VA_ARGS__) APPLY_PO_9(9, __VA_ARGS__)
#define APPLY_PO_STAGE_11(...) APPLY_PO_STAGE_10(__VA_ARGS__) APPLY_PO_10(10, __VA_ARGS__)
#define APPLY_PO_STAGE_12(...) APPLY_PO_STAGE_11(__VA_ARGS__) APPLY_PO_11(11, __VA_ARGS__)
#define APPLY_PO_STAGE_13(...) APPLY_PO_STAGE_12(__VA_ARGS__) APPLY_PO_12(12, __VA_ARGS__)
#define APPLY_PO_STAGE_14(...) APPLY_PO_STAGE_13(__VA_ARGS__) APPLY_PO_13(13, __VA_ARGS__)
#define APPLY_PO_STAGE_15(...) APPLY_PO_STAGE_14(__VA_ARGS__) APPLY_PO_14(14, __VA_ARGS__)
#define APPLY_PO_STAGE_16(...) APPLY_PO_STAGE_15(__VA_ARGS__) APPLY_PO_15(15, __VA_ARGS__)
#define APPLY_PO_STAGE_17(...) APPLY_PO_STAGE_16(__VA_ARGS__) APPLY_PO_16(16, __VA_ARGS__)
#define APPLY_PO_STAGE_18(...) APPLY_PO_STAGE_17(__VA_ARGS__) APPLY_PO_17(17, __VA_ARGS__)
#define APPLY_PO_STAGE_19(...) APPLY_PO_STAGE_18(__VA_ARGS__) APPLY_PO_18(18, __VA_ARGS__)
#define APPLY_PO_STAGE_20(...) APPLY_PO_STAGE_19(__VA_ARGS__) APPLY_PO_19(19, __VA_ARGS__)
#define APPLY_PO_STAGE_21(...) APPLY_PO_STAGE_20(__VA_ARGS__) APPLY_PO_20(20, __VA_ARGS__)
#define APPLY_PO_STAGE_22(...) APPLY_PO_STAGE_21(__VA_ARGS__) APPLY_PO_21(21, __VA_ARGS__)
#define APPLY_PO_STAGE_23(...) APPLY_PO_STAGE_22(__VA_ARGS__) APPLY_PO_22(22, __VA_ARGS__)
#define APPLY_PO_STAGE_24(...) APPLY_PO_STAGE_23(__VA_ARGS__) APPLY_PO_23(23, __VA_ARGS__)
#define APPLY_PO_STAGE_25(...) APPLY_PO_STAGE_24(__VA_ARGS__) APPLY_PO_24(24, __VA_ARGS__)
#define APPLY_PO_STAGE_26(...) APPLY_PO_STAGE_25(__VA_ARGS__) APPLY_PO_25(25, __VA_ARGS__)
#define APPLY_PO_STAGE_27(...) APPLY_PO_STAGE_26(__VA_ARGS__) APPLY_PO_26(26, __VA_ARGS__)
#define APPLY_PO_STAGE_28(...) APPLY_PO_STAGE_27(__VA_ARGS__) APPLY_PO_27(27, __VA_ARGS__)
#define APPLY_PO_STAGE_29(...) APPLY_PO_STAGE_28(__VA_ARGS__) APPLY_PO_28(28, __VA_ARGS__)
#define APPLY_PO_STAGE_30(...) APPLY_PO_STAGE_29(__VA_ARGS__) APPLY_PO_29(29, __VA_ARGS__)
#define APPLY_PO_STAGE_31(...) APPLY_PO_STAGE_30(__VA_ARGS__) APPLY_PO_30(30, __VA_ARGS__)
#define APPLY_PO_STAGE_32(...) APPLY_PO_STAGE_31(__VA_ARGS__) APPLY_PO_31(31, __VA_ARGS__)
// clang-format on

#define APPLY_POST_OPS_SERIAL(result, sum_src, x0, x1, x2, x3, x4, x5) \
    { \
        POST_OP_DATA_T acc; \
        write(&acc, result); \
        CONCAT2(APPLY_PO_STAGE_, POST_OP_CHAIN_LENGTH) \
        (acc, sum_src, x0, x1, x2, x3, x4, x5); \
        write(&result, acc); \
    }

#else

#define APPLY_POST_OPS_SERIAL(...)

#endif // WITH_POST_OP

#endif
