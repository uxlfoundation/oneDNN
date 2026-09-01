//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<float>, \
    size_t, size_t, \
    const float *, \
    OutputArg<float>, \
    const float *, Activation, bool

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_hybrid_fp32_mla_4x24( ARGLIST );
void a64_hybrid_fp32_mla_4x24_a55( ARGLIST );

class cls_a64_hybrid_fp32_mla_4x24
{
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 4;
    }

    static unsigned int out_width()
    {
        return 24;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 4, 24, 1> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 2.985 };
                case CPUModel::A53:
                    return { 1.43 };
                case CPUModel::A73:
                    return { 2.56 };
                case CPUModel::A510:
                    return { 3.51 };
                case CPUModel::V1:
                    return { 13.86 };
                default:
                    return { 6.614 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_hybrid_fp32_mla_4x24;
    cls_a64_hybrid_fp32_mla_4x24(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
            case CPUModel::A53:
                kernel=a64_hybrid_fp32_mla_4x24_a55;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
