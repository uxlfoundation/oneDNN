//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __aarch64__

#include "../std_transforms_fixed_trB.hpp"
#include "../performance_parameters.hpp"

#include "kai/ops/bfloat.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_sgemm_asimd_8x12(const float *, const float *, float *, int, int, int);
void a64_sgemm_asimd_8x12_a53(const float *, const float *, float *, int, int, int);
void a64_sgemm_asimd_8x12_a55(const float *, const float *, float *, int, int, int);
void a64_sgemm_asimd_8x12_a55r1(const float *, const float *, float *, int, int, int);
void a64_sgemm_asimd_8x12_x1(const float *, const float *, float *, int, int, int);

// 8x12 SGEMM "strategy" class.
//
// This describes the characteristics of a family of kernels, in terms of
// the required interleave properties and the output block size.
//
// All kernels in the family must share these characteristics.  The actual
// kernel to be used can be chosen at runtime, based on the CPU_type
// structure.
class cls_a64_sgemm_8x12 {
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, const float *, float *, int, int, int);

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
    StdTransformsFixedTRB<lhs_operand_type, rhs_operand_type, result_type, 8, 12> transforms = {};

    template<typename T>
    static PerformanceParameters get_performance_parameters(const CPUInfo *ci) {
        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 3.954, 1.252, 1.141 };

                case CPUModel::A53:
                    return { 2.777, 0.987, 0.898 };

                case CPUModel::A73:
                    return { 2.885, 1.429, 1.163 };

                case CPUModel::V1:
                    return { 14.95, 9.95, 5.28 };

                default:
                    return { 7.2307, 3.876, 2.932 };
            }
        }

        if (std::is_same<T, bfloat16>::value) {
            switch(ci->get_cpu_model()) {
                case CPUModel::A510:
                    return { 4.98, 2.27, 3.05 };

                default:
                    return { 7.99, 5.06, 7.32 };
            }
        }

        return { 1.0 };
    }

    kern_type kernel=a64_sgemm_asimd_8x12;

    cls_a64_sgemm_8x12(const CPUInfo *ci) {
        // Select specific kernel if available
        switch(ci->get_cpu_model()) {
            case CPUModel::A53:
                kernel = a64_sgemm_asimd_8x12_a53;
                break;

            case CPUModel::A55r0:
                kernel = a64_sgemm_asimd_8x12_a55;
                break;

            case CPUModel::A55r1:
                kernel = a64_sgemm_asimd_8x12_a55r1;
                break;

            case CPUModel::X1:
                kernel = a64_sgemm_asimd_8x12_x1;
                break;

            default:
                /* Generic kernel is initialized by default. */
                break;
        }
    }
};

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
