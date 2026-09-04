//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sme.hpp"

#define ARGLIST  \
    const float *, const float *, \
    float *, size_t, size_t, \
    const float *, Activation, bool

namespace kai {
namespace ops {

void sme_gemv_fp32_mla_8VL( ARGLIST );

class cls_sme_gemv_fp32_mla_8VL
{
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)( ARGLIST );

    static unsigned int out_width()
    {
        return sme::get_vector_length<float>() * 8;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
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


    StdTransformsSME<operand_type, result_type, 1, 8, 1> transforms = {};


    // Default to the generic kernel
    kern_type kernel=sme_gemv_fp32_mla_8VL;
    cls_sme_gemv_fp32_mla_8VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
