//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../kernel_weight_format.hpp"
#include "../performance_parameters.hpp"
#include "../std_transforms_sve.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<__fp16>, \
    size_t, size_t, \
    const __fp16 *, \
    size_t, \
    OutputArg<__fp16>, \
    const __fp16 *, Activation, bool

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_ffhybrid_fp16_mla_6x4VL( ARGLIST );
void sve_ffhybrid_fp16_mla_6x4VL_a64fx( ARGLIST );

class cls_sve_ffhybrid_fp16_mla_6x4VL
{
public:
    typedef __fp16 lhs_operand_type;
    typedef __fp16 rhs_operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }
    static unsigned int stripe_width()
    {
        return get_vector_length<__fp16>() * 1;
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL1VL_BL16;
    }

    static unsigned int out_width()
    {
        return get_vector_length<__fp16>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 4, 1> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, __fp16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.51 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_ffhybrid_fp16_mla_6x4VL;
    cls_sve_ffhybrid_fp16_mla_6x4VL(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A64FX:
                kernel=sve_ffhybrid_fp16_mla_6x4VL_a64fx;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
