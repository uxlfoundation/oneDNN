//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
extern "C" void kai_ops_a64_hybrid_fp32_mla_8x4( const KernelArgs * );
extern "C" void kai_ops_a64_hybrid_fp32_mla_8x4_a55( const KernelArgs * );

class cls_a64_hybrid_fp32_mla_8x4
{
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)( const KernelArgs * );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 4, 4, 1> transforms = {};

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_hybrid_fp32_mla_8x4;
    cls_a64_hybrid_fp32_mla_8x4(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
            case CPUModel::A53:
                kernel=kai_ops_a64_hybrid_fp32_mla_8x4_a55;
                break;
        }
    }
};

} // namespace ops
} // namespace kai
