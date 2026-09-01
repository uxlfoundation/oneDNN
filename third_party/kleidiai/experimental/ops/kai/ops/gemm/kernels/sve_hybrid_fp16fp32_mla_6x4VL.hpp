//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sve.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<__fp16>, \
    size_t, size_t, \
    const __fp16 *, \
    OutputArg<float>, \
    const float *, Activation, bool

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_hybrid_fp16fp32_mla_6x4VL( ARGLIST );

class cls_sve_hybrid_fp16fp32_mla_6x4VL
{
public:
    typedef __fp16 lhs_operand_type;
    typedef __fp16 rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
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
    kern_type kernel=sve_hybrid_fp16fp32_mla_6x4VL;
    cls_sve_hybrid_fp16fp32_mla_6x4VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
