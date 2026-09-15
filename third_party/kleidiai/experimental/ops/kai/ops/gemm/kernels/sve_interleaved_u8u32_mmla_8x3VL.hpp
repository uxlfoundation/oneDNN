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
void sve_interleaved_u8u32_mmla_8x3VL( ARGLIST );

class cls_sve_interleaved_u8u32_mmla_8x3VL
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
        return 8;
    }


    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 6, 8, 2> transforms = {};
    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 8, 6, 8, 2, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, uint32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 61.97, 4.11, 7.93 };
                case CPUModel::A510:
                    return { 43.18, 3.57, 2.89 };
                case CPUModel::V1:
                    return { 123.47, 5.03, 11.76 };
            }
        }


        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 62.00, 4.08, 0.51 };
                case CPUModel::A510:
                    return { 38.02, 1.85, 0.28 };
                case CPUModel::V1:
                    return { 95.28, 7.99, 0.79 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_interleaved_u8u32_mmla_8x3VL;
    cls_sve_interleaved_u8u32_mmla_8x3VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
