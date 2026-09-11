//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<uint8_t>, \
    size_t, size_t, \
    const uint8_t *, \
    OutputArg<uint32_t>, \
    const uint32_t *, Activation, bool

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_hybrid_u8u32_dot_6x16( ARGLIST );
void a64_hybrid_u8u32_dot_6x16_a55( ARGLIST );

class cls_a64_hybrid_u8u32_dot_6x16
{
public:
    typedef uint8_t lhs_operand_type;
    typedef uint8_t rhs_operand_type;
    typedef uint32_t result_type;

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
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 6, 16, 4> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, uint32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.63 };
                case CPUModel::A510:
                    return { 15.89 };
                case CPUModel::V1:
                    return { 53.87 };
                case CPUModel::A55r1:
                    return { 9.217 };
            }
        }

        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 9.5238, 2.0799, 0.2279 };
                default:
                    return { 29.6736, 11.4025, 0.5591 };
                case CPUModel::A510:
                    return { 16.65, 3.92, 0.48 };
                case CPUModel::V1:
                    return { 42.62, 16.32, 0.83 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_hybrid_u8u32_dot_6x16;
    cls_a64_hybrid_u8u32_dot_6x16(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
                kernel=a64_hybrid_u8u32_dot_6x16_a55;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
