//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sve.hpp"
#include "kai/ops/bfloat.hpp"
#include "../kernel_weight_format.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const bfloat16 *, const bfloat16 *, size_t, \
    float *, int, size_t, int

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_ffinterleaved_bf16fp32_mmla_8x3VL( ARGLIST );

class cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL
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
    static unsigned int stripe_width()
    {
        return get_vector_length<float>();
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL2VL_BL64;
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
                    return { 39.90, 8.55, 4.42 };
            }
        }


        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 39.66, 5.18, 4.37 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_ffinterleaved_bf16fp32_mmla_8x3VL;
    cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
