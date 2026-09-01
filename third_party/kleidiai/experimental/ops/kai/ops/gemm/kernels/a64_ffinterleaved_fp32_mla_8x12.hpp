//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_fixed.hpp"
#include "../kernel_weight_format.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const float *, const float *, size_t, \
    float *, int, size_t, int

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_ffinterleaved_fp32_mla_8x12( ARGLIST );

class cls_a64_ffinterleaved_fp32_mla_8x12
{
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
    typedef float result_type;

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
    static unsigned int stripe_width()
    {
        return 4;
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL128_BL32;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }


    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 1> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 1, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 12.56, 9.83, 3.02 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_ffinterleaved_fp32_mla_8x12;
    cls_a64_ffinterleaved_fp32_mla_8x12(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
