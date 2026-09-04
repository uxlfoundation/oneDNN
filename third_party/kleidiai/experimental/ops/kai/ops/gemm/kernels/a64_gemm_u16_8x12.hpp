//
// SPDX-FileCopyrightText: Copyright 2017-2020, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __aarch64__

#include "../std_transforms_fixed.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_gemm_u16_asimd_8x12(const uint16_t *, const uint16_t *, uint32_t *, int, int, int);

class cls_a64_gemm_u16_8x12 {
public:
    typedef uint16_t lhs_operand_type;
    typedef uint16_t rhs_operand_type;
    typedef uint32_t result_type;

    typedef void (*kern_type)(const uint16_t *, const uint16_t *, uint32_t *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width() {
        return 12;
    }

    static unsigned int out_height() {
        return 8;
    }

    static unsigned int k_unroll() {
        return 1;
    }

    // Use the standard fixed size transforms.
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 1, true> transforms_quantized = {};

    kern_type kernel = a64_gemm_u16_asimd_8x12;

    cls_a64_gemm_u16_8x12(const CPUInfo *) { }
};

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
