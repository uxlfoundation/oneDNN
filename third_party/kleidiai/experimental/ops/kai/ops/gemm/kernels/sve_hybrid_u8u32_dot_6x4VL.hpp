//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../performance_parameters.hpp"
#include "../std_transforms_sve.hpp"

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
void sve_hybrid_u8u32_dot_6x4VL( ARGLIST );
void sve_hybrid_u8u32_dot_6x4VL_a64fx( ARGLIST );

class cls_sve_hybrid_u8u32_dot_6x4VL
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
        return get_vector_length<uint32_t>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 4, 4> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, uint32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.56 };
                case CPUModel::A510:
                    return { 20.98 };
                case CPUModel::V1:
                    return { 62.19 };
                case CPUModel::A64FX:
                    return { 91.23 };
            }
        }

        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.59, 15.67, 0.61 };
                case CPUModel::A510:
                    return { 22.75, 3.90, 0.47 };
                case CPUModel::V1:
                    return { 48.09, 16.24, 0.83 };
                case CPUModel::A64FX:
                    return { 101.62, 3.15, 0.42 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_hybrid_u8u32_dot_6x4VL;
    cls_sve_hybrid_u8u32_dot_6x4VL(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A64FX:
                kernel=sve_hybrid_u8u32_dot_6x4VL_a64fx;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
