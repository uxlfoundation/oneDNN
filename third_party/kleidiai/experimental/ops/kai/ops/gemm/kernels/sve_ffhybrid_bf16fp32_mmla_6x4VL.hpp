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
#include "kai/ops/bfloat.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<bfloat16>, \
    size_t, size_t, \
    const bfloat16 *, \
    size_t, \
    OutputArg<float>, \
    const float *, Activation, bool

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_ffhybrid_bf16fp32_mmla_6x4VL( ARGLIST );

class cls_sve_ffhybrid_bf16fp32_mmla_6x4VL
{
public:
    typedef bfloat16 lhs_operand_type;
    typedef bfloat16 rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }
    static unsigned int stripe_width()
    {
        return get_vector_length<float>() * 1;
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL2VL_BL64;
    }

    static unsigned int out_width()
    {
        return get_vector_length<float>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 8, 4> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, bfloat16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 49.10 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_ffhybrid_bf16fp32_mmla_6x4VL;
    cls_sve_ffhybrid_bf16fp32_mmla_6x4VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
