//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sme.hpp"

#define ARGLIST  \
    const int8_t *, const int8_t *, \
    int8_t *, size_t, size_t, \
    const Requantize32 *, const int32_t *, unsigned int

namespace kai {
namespace ops {

void sme_gemv_s8qa_dot_8VL( ARGLIST );

class cls_sme_gemv_s8qa_dot_8VL
{
public:
    typedef int8_t operand_type;
    typedef int8_t result_type;

    typedef void (*kern_type)( ARGLIST );

    static unsigned int out_width()
    {
        return sme::get_vector_length<int32_t>() * 8;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    static constexpr bool supports_bias()
    {
        return false;
    }

    static constexpr bool supports_activation()
    {
        return false;
    }


    StdTransformsSME<operand_type, result_type, 1, 8, 4> transforms = {};


    // Default to the generic kernel
    kern_type kernel=sme_gemv_s8qa_dot_8VL;
    cls_sme_gemv_s8qa_dot_8VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
