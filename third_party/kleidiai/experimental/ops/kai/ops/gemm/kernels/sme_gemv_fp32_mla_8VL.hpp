//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include "common_internal/kernel_args.hpp"
#include "../std_transforms_sme.hpp"

namespace kai {
namespace ops {

extern "C" void kai_ops_sme_gemv_fp32_mla_8VL( const KernelArgs * );

class cls_sme_gemv_fp32_mla_8VL
{
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)( const KernelArgs * );

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
    kern_type kernel=kai_ops_sme_gemv_fp32_mla_8VL;
    cls_sme_gemv_fp32_mla_8VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
