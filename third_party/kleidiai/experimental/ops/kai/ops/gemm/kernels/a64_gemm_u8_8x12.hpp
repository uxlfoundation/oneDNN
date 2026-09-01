//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2021, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __aarch64__

#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

namespace kai {
namespace ops {

// Load the actual kernel
void a64_gemm_u8_8x12(const uint8_t *, const uint8_t *, uint32_t *, int, int, int);
void a64_gemm_u8_8x12_a55r1(const uint8_t *, const uint8_t *, uint32_t *, int, int, int);
void a64_gemm_u8_8x12_x1(const uint8_t *, const uint8_t *, uint32_t *, int, int, int);

class cls_a64_gemm_u8_8x12 {
public:
    typedef uint8_t lhs_operand_type;
    typedef uint8_t rhs_operand_type;
    typedef uint32_t result_type;

    typedef void (*kern_type)(const uint8_t *, const uint8_t *, uint32_t *, int, int, int);

    /* Describes the data layout for A input */
    static const int A_interleave = 8;
    static const int A_block = 4;
    static const bool A_transpose = false;

    /* Same for B input */
    static const int B_interleave = 12;
    static const int B_block = 4;
    static const bool B_transpose = true;

    /* Kernel blocking parameters */
    static unsigned int out_width() {
        return 12;
    }

    static unsigned int out_height() {
        return 8;
    }

    static unsigned int k_unroll() {
        return 4;
    }

    // Use the standard fixed sized transforms.
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 4> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 4, true> transforms_quantized = {};

    template<typename T>
    static PerformanceParameters get_performance_parameters(const CPUInfo *ci) {
        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A510:
                    return { 19.73, 3.38, 0.27 };

                case CPUModel::A55r1:
                    return { 15.361, 0.9341, 0.1636 };

                case CPUModel::V1:
                    return { 51.14, 7.38, 0.65 };

                default:
                    return { 29.0698, 3.9793, 0.4003 };
            }
        }

        if (std::is_same<T, uint32_t>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A510:
                    return { 19.73, 3.38, 3.70 };

                case CPUModel::A55r1:
                    return { 14.286, 1.171, 1.209 };

                case CPUModel::V1:
                    return { 61.58, 4.78, 10.83 };

                default:
                    return { 31.82, 3.51, 8.03 };
            }
        }

        return { 0.0 };
    }

    kern_type kernel = a64_gemm_u8_8x12;

    cls_a64_gemm_u8_8x12(const CPUInfo *ci) {
        auto mod = ci->get_cpu_model();

        if (mod == CPUModel::A55r1) {
            kernel = a64_gemm_u8_8x12_a55r1;
        } else if (mod == CPUModel::X1) {
            kernel = a64_gemm_u8_8x12_x1;
        }
    }
};

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
