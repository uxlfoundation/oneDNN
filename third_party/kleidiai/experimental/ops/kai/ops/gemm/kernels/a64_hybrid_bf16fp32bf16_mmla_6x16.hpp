//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include "common_internal/kernel_args.hpp"
#include "../std_transforms_fixed.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
extern "C" void kai_ops_a64_hybrid_bf16fp32bf16_mmla_6x16( const KernelArgs * );

class cls_a64_hybrid_bf16fp32bf16_mmla_6x16
{
public:
    typedef bfloat16 lhs_operand_type;
    typedef bfloat16 rhs_operand_type;
    typedef bfloat16 result_type;

    typedef void (*kern_type)( const KernelArgs * );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }

    static unsigned int out_width()
    {
        return 16;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 6, 16, 16, 4> transforms = {};

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_hybrid_bf16fp32bf16_mmla_6x16;
    cls_a64_hybrid_bf16fp32bf16_mmla_6x16(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai
