//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
extern "C" void kai_ops_sve_hybrid_s8qs_mmla_6x4VL( const KernelArgs * );

class cls_sve_hybrid_s8qs_mmla_6x4VL
{
public:
    typedef int8_t lhs_operand_type;
    typedef int8_t rhs_operand_type;
    typedef int8_t result_type;

    typedef void (*kern_type)( const KernelArgs * );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }

    static unsigned int out_width()
    {
        return get_vector_length<int32_t>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 8;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 8, 8, 8> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, int8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 49.98 };
                case CPUModel::A510:
                    return { 22.62 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_sve_hybrid_s8qs_mmla_6x4VL;
    cls_sve_hybrid_s8qs_mmla_6x4VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai
