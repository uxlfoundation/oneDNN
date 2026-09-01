//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __arm__

#include "../std_transforms_fixed.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
void a32_sgemm_8x6(const float *, const float *, float *, int, int, int);
void a32_sgemm_8x6_a53(const float *, const float *, float *, int, int, int);
void a32_sgemm_8x6_a55r1(const float *, const float *, float *, int, int, int);

// 8x6 SGEMM "strategy" class.
//
// This describes the characteristics of a family of kernels, in terms of
// the required interleave properties and the output block size.
//
// All kernels in the family must share these characteristics.  The actual
// kernel to be used can be chosen at runtime, based on the CPU_type
// structure.
class cls_a32_sgemm_8x6 {
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, const float *, float *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width() {
        return 8;
    }

    static unsigned int out_height() {
        return 6;
    }

    static unsigned int k_unroll() {
        return 1;
    }

    // Use the standard fixed size transforms.
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 6, 8> transforms = {};

    kern_type kernel = a32_sgemm_8x6;

    cls_a32_sgemm_8x6(const CPUInfo *ci) {
        switch(ci->get_cpu_model()) {
            case CPUModel::A53:
                kernel = a32_sgemm_8x6_a53;
                break;

            case CPUModel::A55r1:
                kernel = a32_sgemm_8x6_a55r1;
                break;

            default:
                /* Generic kernel is selected by default. */
                break;
        }
    }
};

}  // namespace ops
}  // namespace kai
#endif // __arm__
