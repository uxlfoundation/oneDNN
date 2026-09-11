//
// SPDX-FileCopyrightText: Copyright 2019-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sve.hpp"
#include "kai/ops/bfloat.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const bfloat16 *, const bfloat16 *, \
    float *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_interleaved_bf16fp32_mmla_8x3VL( ARGLIST );

class cls_sve_interleaved_bf16fp32_mmla_8x3VL
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
        return get_vector_length<float>() * 3;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }


    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 6, 4, 2> transforms = {};
    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 6, 4, 2, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, bfloat16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.41, 4.30, 7.14 };
                case CPUModel::A510:
                    return { 7.78, 4.01, 2.43 };
                case CPUModel::V1:
                    return { 62.50, 5.09, 11.32 };
            }
        }


        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 30.86, 2.36, 5.28 };
                case CPUModel::A510:
                    return { 7.75, 2.47, 2.39 };
                case CPUModel::V1:
                    return { 47.63, 5.11, 6.80 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_interleaved_bf16fp32_mmla_8x3VL;
    cls_sve_interleaved_bf16fp32_mmla_8x3VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
