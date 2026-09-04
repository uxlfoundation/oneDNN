//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sve.hpp"
#include "../kernel_weight_format.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const __fp16 *, const __fp16 *, size_t, \
    __fp16 *, int, size_t, int

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_ffinterleaved_fp16_mla_8x3VL( ARGLIST );
void sve_ffinterleaved_fp16_mla_8x3VL_a64fx( ARGLIST );

class cls_sve_ffinterleaved_fp16_mla_8x3VL
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
    static unsigned int stripe_width()
    {
        return get_vector_length<__fp16>();
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL1VL_BL16;
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
                    return { 25.53, 7.89, 3.82 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_ffinterleaved_fp16_mla_8x3VL;
    cls_sve_ffinterleaved_fp16_mla_8x3VL(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A64FX:
                kernel=sve_ffinterleaved_fp16_mla_8x3VL_a64fx;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
