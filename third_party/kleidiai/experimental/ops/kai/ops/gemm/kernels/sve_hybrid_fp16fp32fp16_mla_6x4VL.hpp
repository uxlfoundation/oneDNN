//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include "common_internal/kernel_args.hpp"
#include "../std_transforms_sve.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
extern "C" void kai_ops_sve_hybrid_fp16fp32fp16_mla_6x4VL( const KernelArgs * );

class cls_sve_hybrid_fp16fp32fp16_mla_6x4VL
{
public:
    typedef __fp16 lhs_operand_type;
    typedef __fp16 rhs_operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( const KernelArgs * );

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

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 2, 2, 1> transforms = {};

    // Default to the generic kernel
    kern_type kernel=kai_ops_sve_hybrid_fp16fp32fp16_mla_6x4VL;
    cls_sve_hybrid_fp16fp32fp16_mla_6x4VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai
