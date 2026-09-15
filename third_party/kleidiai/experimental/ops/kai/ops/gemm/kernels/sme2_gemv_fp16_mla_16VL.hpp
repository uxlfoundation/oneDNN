//
// SPDX-FileCopyrightText: Copyright 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sme.hpp"

#define ARGLIST  \
    const __fp16 *, const __fp16 *, \
    __fp16 *, size_t, size_t, \
    const __fp16 *, Activation, bool

namespace kai {
namespace ops {

void sme2_gemv_fp16_mla_16VL( ARGLIST );

class cls_sme2_gemv_fp16_mla_16VL
{
public:
    typedef __fp16 operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( ARGLIST );

    static unsigned int out_width()
    {
        return sme::get_vector_length<__fp16>() * 16;
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


    StdTransformsSME<operand_type, result_type, 1, 16, 1> transforms = {};


    // Default to the generic kernel
    kern_type kernel=sme2_gemv_fp16_mla_16VL;
    cls_sme2_gemv_fp16_mla_16VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
