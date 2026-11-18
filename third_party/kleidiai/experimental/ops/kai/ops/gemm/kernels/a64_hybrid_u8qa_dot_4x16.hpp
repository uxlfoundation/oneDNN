//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
extern "C" void kai_ops_a64_hybrid_u8qa_dot_4x16( const KernelArgs * );
extern "C" void kai_ops_a64_hybrid_u8qa_dot_4x16_a55( const KernelArgs * );

class cls_a64_hybrid_u8qa_dot_4x16
{
public:
    typedef uint8_t lhs_operand_type;
    typedef uint8_t rhs_operand_type;
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
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 4, 16, 16, 4> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 7.5301 };
                default:
                    return { 27.5482 };
                case CPUModel::A510:
                    return { 14.81 };
                case CPUModel::V1:
                    return { 44.54 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_hybrid_u8qa_dot_4x16;
    cls_a64_hybrid_u8qa_dot_4x16(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
                kernel=kai_ops_a64_hybrid_u8qa_dot_4x16_a55;
                break;
        }
    }
};

} // namespace ops
} // namespace kai
