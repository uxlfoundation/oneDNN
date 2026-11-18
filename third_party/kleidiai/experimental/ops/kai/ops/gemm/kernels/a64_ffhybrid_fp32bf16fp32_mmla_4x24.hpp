//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include "common_internal/kernel_args.hpp"
#include "../kernel_weight_format.hpp"
#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
extern "C" void kai_ops_a64_ffhybrid_fp32bf16fp32_mmla_4x24( const KernelArgs * );

class cls_a64_ffhybrid_fp32bf16fp32_mmla_4x24
{
public:
    typedef float lhs_operand_type;
    typedef bfloat16 rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)( const KernelArgs * );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 4;
    }
    static unsigned int stripe_width()
    {
        return 4;
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL256_BL64_BF16;
    }

    static unsigned int out_width()
    {
        return 24;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 4, 4, 4, 4> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 28.48 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_ffhybrid_fp32bf16fp32_mmla_4x24;
    cls_a64_ffhybrid_fp32bf16fp32_mmla_4x24(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai
