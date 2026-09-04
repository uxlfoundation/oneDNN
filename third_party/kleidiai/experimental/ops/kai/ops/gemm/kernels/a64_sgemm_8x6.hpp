//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __aarch64__

#include "../std_transforms_fixed.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_sgemm_asimd_8x6(const float *, const float *, float *, int, int, int);

// 8x6 SGEMM "strategy" class.
//
// This describes the characteristics of a family of kernels, in terms of
// the required interleave properties and the output block size.
//
// All kernels in the family must share these characteristics.  The actual
// kernel to be used can be chosen at runtime, based on the CPU_type
// structure.
class cls_a64_sgemm_8x6 {
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, const float *, float *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width() {
        return 6;
    }

    static unsigned int out_height() {
        return 8;
    }

    static unsigned int k_unroll() {
        return 1;
    }

    // Use the standard fixed size transforms.
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 6, 1> transforms = {};

    kern_type kernel=a64_sgemm_asimd_8x6;

    cls_a64_sgemm_8x6(const CPUInfo *) {

    }
};

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
