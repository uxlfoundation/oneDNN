//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sve.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const float *, const float *, \
    float *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_interleaved_fp32_mla_8x3VL( ARGLIST );
void sve_interleaved_fp32_mla_8x3VL_a64fx( ARGLIST );

class cls_sve_interleaved_fp32_mla_8x3VL
{
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
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
        return 1;
    }


    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 3, 1, 1> transforms = {};
    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 3, 1, 1, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 7.2307, 3.876, 2.932 };
                case CPUModel::A64FX:
                    return { 26.52, 3.42, 4.59 };
                case CPUModel::A510:
                    return { 6.25, 3.84, 2.47 };
                case CPUModel::V1:
                    return { 15.15, 9.24, 6.42 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_interleaved_fp32_mla_8x3VL;
    cls_sve_interleaved_fp32_mla_8x3VL(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A64FX:
                kernel=sve_interleaved_fp32_mla_8x3VL_a64fx;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
