//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sme.hpp"
#include "kai/ops/bfloat.hpp"

#define ARGLIST  \
    const float *, const bfloat16 *, \
    float *, size_t, size_t, \
    const float *, Activation, bool

namespace kai {
namespace ops {

void sme_gemv_fp32bf16fp32_dot_8VL( ARGLIST );

class cls_sme_gemv_fp32bf16fp32_dot_8VL
{
public:
    typedef bfloat16 operand_type;
    typedef float result_type;

    typedef void (*kern_type)( ARGLIST );

    static unsigned int out_width()
    {
        return sme::get_vector_length<float>() * 8;
    }

    static constexpr unsigned int k_unroll()
    {
        return 2;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    static constexpr bool supports_bias()
    {
        return true;
    }

    static constexpr bool supports_activation()
    {
        return true;
    }


    StdTransformsSME<operand_type, result_type, 1, 8, 2> transforms = {};


    // Default to the generic kernel
    kern_type kernel=sme_gemv_fp32bf16fp32_dot_8VL;
    cls_sme_gemv_fp32bf16fp32_dot_8VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
