//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../kernel_weight_format.hpp"
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
void sve_ffhybrid_fp16fp32fp16_mla_6x4VL( ARGLIST );

class cls_sve_ffhybrid_fp16fp32fp16_mla_6x4VL
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
        return get_vector_length<float>() * 2;
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL1VL_BL16;
    }

    static unsigned int out_width()
    {
        return get_vector_length<float>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 2, 1> transforms = {};

    // Default to the generic kernel
    kern_type kernel=sve_ffhybrid_fp16fp32fp16_mla_6x4VL;
    cls_sve_ffhybrid_fp16fp32fp16_mla_6x4VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
