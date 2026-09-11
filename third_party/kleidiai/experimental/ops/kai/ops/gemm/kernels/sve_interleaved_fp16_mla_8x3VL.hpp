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
    const __fp16 *, const __fp16 *, \
    __fp16 *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_interleaved_fp16_mla_8x3VL( ARGLIST );
void sve_interleaved_fp16_mla_8x3VL_a64fx( ARGLIST );

class cls_sve_interleaved_fp16_mla_8x3VL
{
public:
    typedef __fp16 lhs_operand_type;
    typedef __fp16 rhs_operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return get_vector_length<__fp16>() * 3;
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

        if (std::is_same<T, __fp16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 15.96, 3.85, 6.91 };
                case CPUModel::A510:
                    return { 13.84, 2.07, 2.52 };
                case CPUModel::V1:
                    return { 31.90, 5.15, 10.34 };
                case CPUModel::A64FX:
                    return { 44.34, 3.23, 7.06 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_interleaved_fp16_mla_8x3VL;
    cls_sve_interleaved_fp16_mla_8x3VL(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A64FX:
                kernel=sve_interleaved_fp16_mla_8x3VL_a64fx;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
