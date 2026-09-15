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
    const int8_t *, const int8_t *, \
    int32_t *, int, int, int

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_interleaved_s8s32_mmla_8x3VL( ARGLIST );

class cls_sve_interleaved_s8s32_mmla_8x3VL
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
        return get_vector_length<int32_t>() * 3;
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

        if (std::is_same<T, int32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 61.98, 3.90, 7.94 };
                case CPUModel::V1:
                    return { 123.42, 5.00, 11.52 };
                case CPUModel::A510:
                    return { 43.14, 3.62, 2.90 };
            }
        }


        if (std::is_same<T, int8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 61.97, 3.64, 0.50 };
                case CPUModel::V1:
                    return { 95.28, 7.99, 0.79 };
                case CPUModel::A510:
                    return { 43.36, 1.86, 0.28 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_interleaved_s8s32_mmla_8x3VL;
    cls_sve_interleaved_s8s32_mmla_8x3VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
