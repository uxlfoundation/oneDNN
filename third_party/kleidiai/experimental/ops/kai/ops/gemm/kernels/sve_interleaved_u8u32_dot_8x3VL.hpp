//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_sve.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const uint8_t *, const uint8_t *, \
    uint32_t *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_interleaved_u8u32_dot_8x3VL( ARGLIST );
void sve_interleaved_u8u32_dot_8x3VL_a64fx( ARGLIST );

class cls_sve_interleaved_u8u32_dot_8x3VL
{
public:
    typedef uint8_t lhs_operand_type;
    typedef uint8_t rhs_operand_type;
    typedef uint32_t result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return get_vector_length<uint32_t>() * 3;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }


    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 3, 4, 1> transforms = {};
    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 3, 4, 1, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, uint32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.66, 4.11, 7.94 };
                case CPUModel::A510:
                    return { 27.44, 3.41, 2.90 };
                case CPUModel::V1:
                    return { 63.30, 4.97, 11.52 };
                case CPUModel::A64FX:
                    return { 109.76, 3.88, 6.76 };
            }
        }


        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.67, 4.04, 0.50 };
                case CPUModel::A510:
                    return { 27.45, 1.65, 0.28 };
                case CPUModel::V1:
                    return { 52.24, 7.49, 0.80 };
                case CPUModel::A64FX:
                    return { 110.18, 2.34, 0.40 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_interleaved_u8u32_dot_8x3VL;
    cls_sve_interleaved_u8u32_dot_8x3VL(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A64FX:
                kernel=sve_interleaved_u8u32_dot_8x3VL_a64fx;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
