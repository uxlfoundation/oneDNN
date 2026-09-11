//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<__fp16>, \
    size_t, size_t, \
    const __fp16 *, \
    OutputArg<__fp16>, \
    const __fp16 *, Activation, bool

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_hybrid_fp16_mla_6x32( ARGLIST );
void a64_hybrid_fp16_mla_6x32_a55( ARGLIST );

class cls_a64_hybrid_fp16_mla_6x32
{
public:
    typedef __fp16 lhs_operand_type;
    typedef __fp16 rhs_operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }

    static unsigned int out_width()
    {
        return 32;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 6, 32, 1> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, __fp16>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 6.94 };
                default:
                    return { 14.53 };
                case CPUModel::A510:
                    return { 8.94 };
                case CPUModel::V1:
                    return { 29.26 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_hybrid_fp16_mla_6x32;
    cls_a64_hybrid_fp16_mla_6x32(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
                kernel=a64_hybrid_fp16_mla_6x32_a55;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
