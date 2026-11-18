//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include "common_internal/kernel_args.hpp"
#include "../performance_parameters.hpp"
#include "../std_transforms_sve.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
extern "C" void kai_ops_sve_hybrid_u8qa_dot_4x4VL( const KernelArgs * );

class cls_sve_hybrid_u8qa_dot_4x4VL
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
        return get_vector_length<uint32_t>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 4, 4, 4, 4> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 29.89 };
                case CPUModel::A510:
                    return { 17.12 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_sve_hybrid_u8qa_dot_4x4VL;
    cls_sve_hybrid_u8qa_dot_4x4VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai
