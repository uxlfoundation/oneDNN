//
// SPDX-FileCopyrightText: Copyright 2017-2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __aarch64__

#include "../std_transforms_fixed.hpp"
#include "../performance_parameters.hpp"

namespace kai {
namespace ops {

// Load the actual kernel
void a64_gemm_s8_4x4(const int8_t *, const int8_t *, int32_t *, int, int, int);

#include "kai/ops/gemm/kai_ops.hpp"

class cls_a64_gemm_s8_4x4 {
public:
    typedef int8_t lhs_operand_type;
    typedef int8_t rhs_operand_type;
    typedef int32_t result_type;

    typedef void (*kern_type)(const int8_t *, const int8_t *, int32_t *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width() {
        return 4;
    }

    static unsigned int out_height() {
        return 4;
    }

    static unsigned int k_unroll() {
        return 16;
    }

    // Use the standard fixed size transforms.
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 4, 4, 16> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 4, 4, 16, true> transforms_quantized = {};

    template<typename T>
    static PerformanceParameters get_performance_parameters(const CPUInfo *ci) {
        if (std::is_same<T, int32_t>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r0:
                case CPUModel::A55r1:
                    return { 3.12, 2.93, 1.84 };
                case CPUModel::A510:
                    return { 3.32, 2.56, 2.63 };
                default:
                    return { 7.97, 3.72, 7.31 };
            }
        }

        if (std::is_same<T, int8_t>::value) {
            switch(ci->get_cpu_model()) {
                case CPUModel::A55r0:
                case CPUModel::A55r1:
                    return { 3.12, 2.18, 0.09 };
                case CPUModel::A510:
                    return { 3.33, 2.89, 0.09 };
                default:
                    return { 7.97, 3.74, 0.34 };
            }
        }

        return { 1.0 };
    }

    kern_type kernel=a64_gemm_s8_4x4;

    cls_a64_gemm_s8_4x4(const CPUInfo *) { }
};

}  // namespace ops
}  // namespace kai

#endif // __aarch64__

