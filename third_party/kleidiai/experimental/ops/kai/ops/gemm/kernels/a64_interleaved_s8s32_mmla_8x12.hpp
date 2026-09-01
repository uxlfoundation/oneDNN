//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_fixed.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const int8_t *, const int8_t *, \
    int32_t *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_interleaved_s8s32_mmla_8x12( ARGLIST );
void a64_interleaved_s8s32_mmla_8x12_a510( ARGLIST );

class cls_a64_interleaved_s8s32_mmla_8x12
{
public:
    typedef int8_t lhs_operand_type;
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

        if (std::is_same<T, int32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 62.57, 4.08, 8.01 };
                case CPUModel::A510:
                    return { 48.25, 3.53, 3.71 };
                case CPUModel::V1:
                    return { 117.02, 4.98, 10.87 };
            }
        }


        if (std::is_same<T, int8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 62.53, 3.70, 0.50 };
                case CPUModel::A510:
                    return { 48.22, 2.49, 0.29 };
                case CPUModel::V1:
                    return { 75.54, 8.06, 0.63 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_interleaved_s8s32_mmla_8x12;
    cls_a64_interleaved_s8s32_mmla_8x12(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A510:
                kernel=a64_interleaved_s8s32_mmla_8x12_a510;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
