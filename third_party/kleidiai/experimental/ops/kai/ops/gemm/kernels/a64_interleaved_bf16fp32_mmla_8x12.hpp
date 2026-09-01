//
// SPDX-FileCopyrightText: Copyright 2019-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_fixed.hpp"
#include "kai/ops/bfloat.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const bfloat16 *, const bfloat16 *, \
    float *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_interleaved_bf16fp32_mmla_8x12( ARGLIST );
void a64_interleaved_bf16fp32_mmla_8x12_a510( ARGLIST );

class cls_a64_interleaved_bf16fp32_mmla_8x12
{
public:
    typedef bfloat16 lhs_operand_type;
    typedef bfloat16 rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return 12;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }


    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 4> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 4, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, bfloat16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.54, 4.30, 7.33 };
                case CPUModel::V1:
                    return { 59.94, 5.08, 9.83 };
                case CPUModel::A510:
                    return { 7.82, 4.05, 3.07 };
            }
        }


        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.15, 2.51, 5.25 };
                case CPUModel::V1:
                    return { 41.44, 5.01, 5.64 };
                case CPUModel::A510:
                    return { 7.83, 2.53, 2.71 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_interleaved_bf16fp32_mmla_8x12;
    cls_a64_interleaved_bf16fp32_mmla_8x12(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A510:
                kernel=a64_interleaved_bf16fp32_mmla_8x12_a510;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
