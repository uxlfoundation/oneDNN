//
// SPDX-FileCopyrightText: Copyright 2019-2020, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sve.hpp"

#define ARGLIST  \
    const float *, const float *, \
    float *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_interleaved_fp32_mmla_8x3VL( ARGLIST );

class cls_sve_interleaved_fp32_mmla_8x3VL
{
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
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

    static constexpr unsigned int k_unroll()
    {
        return 2;
    }


    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 6, 2, 2> transforms = {};
    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 6, 2, 2, true> transforms_quantized = {};

    // Default to the generic kernel
    kern_type kernel=sve_interleaved_fp32_mmla_8x3VL;
    cls_sve_interleaved_fp32_mmla_8x3VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
