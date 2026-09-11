//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<int8_t>, \
    size_t, size_t, \
    const int8_t *, \
    OutputArg<int8_t>, \
    const Requantize32 *, const int32_t *, unsigned int

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_hybrid_s8qs_mmla_6x16( ARGLIST );

class cls_a64_hybrid_s8qs_mmla_6x16
{
public:
    typedef int8_t lhs_operand_type;
    typedef int8_t rhs_operand_type;
    typedef int8_t result_type;

    typedef void (*kern_type)( ARGLIST );

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
        return 8;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 6, 16, 8> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, int8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 50.42 };
                case CPUModel::A510:
                    return { 28.71 };
                case CPUModel::V1:
                    return { 77.72 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_hybrid_s8qs_mmla_6x16;
    cls_a64_hybrid_s8qs_mmla_6x16(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
