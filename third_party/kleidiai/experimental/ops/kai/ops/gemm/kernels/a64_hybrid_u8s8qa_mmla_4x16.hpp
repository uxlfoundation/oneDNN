//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include "common_internal/kernel_args.hpp"
#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
extern "C" void kai_ops_a64_hybrid_u8s8qa_mmla_4x16( const KernelArgs * );

class cls_a64_hybrid_u8s8qa_mmla_4x16
{
public:
    typedef uint8_t lhs_operand_type;
    typedef int8_t rhs_operand_type;
    typedef uint8_t result_type;

    typedef void (*kern_type)( const KernelArgs * );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 4;
    }

    static unsigned int out_width()
    {
        return 16;
    }

    static constexpr unsigned int k_unroll()
    {
        return 8;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 4, 16, 16, 8> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 47.74 };
                case CPUModel::A510:
                    return { 27.99 };
                case CPUModel::V1:
                    return { 62.26 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_hybrid_u8s8qa_mmla_4x16;
    cls_a64_hybrid_u8s8qa_mmla_4x16(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai
