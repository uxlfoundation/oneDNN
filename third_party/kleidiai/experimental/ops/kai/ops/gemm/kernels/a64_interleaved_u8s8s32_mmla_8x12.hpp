//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_fixed.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const uint8_t *, const int8_t *, \
    int32_t *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_interleaved_u8s8s32_mmla_8x12( ARGLIST );

class cls_a64_interleaved_u8s8s32_mmla_8x12
{
public:
    typedef uint8_t lhs_operand_type;
    typedef int8_t rhs_operand_type;
    typedef int32_t result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return 12;
    }

    static constexpr unsigned int k_unroll()
    {
        return 8;
    }


    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 8> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 8, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, uint32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 62.58, 4.06, 8.02 };
                case CPUModel::A510:
                    return { 47.83, 3.59, 3.72 };
                case CPUModel::V1:
                    return { 111.52, 4.97, 10.80 };
            }
        }


        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 62.57, 4.10, 0.51 };
                case CPUModel::A510:
                    return { 47.66, 2.47, 0.29 };
                case CPUModel::V1:
                    return { 75.54, 8.06, 0.63 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_interleaved_u8s8s32_mmla_8x12;
    cls_a64_interleaved_u8s8s32_mmla_8x12(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
