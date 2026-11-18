//
// SPDX-FileCopyrightText: Copyright 2019-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../std_transforms_fixed.hpp"
#include "../performance_parameters.hpp"

namespace kai {
namespace ops {

// Actual kernel implementations
extern "C" void kai_ops_a64_interleaved_bf16fp32_mmla_8x12( const KernelArgs * );
extern "C" void kai_ops_a64_interleaved_bf16fp32_mmla_8x12_a510( const KernelArgs * );

class cls_a64_interleaved_bf16fp32_mmla_8x12
{
public:
    typedef bfloat16 lhs_operand_type;
    typedef bfloat16 rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)( const KernelArgs * );

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
        return 4;
    }

    static constexpr size_t accumulation_buffer_element_size()
    {
        return 0;
    }


    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 12, 4> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 12, 4, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, bfloat16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.54, 4.30, 7.33 };
                case CPUModel::V1:
                    return { 59.94, 5.08, 9.83 };
                case CPUModel::A510:
                    return { 7.82, 4.05, 3.07 };
            }
        }


        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.15, 2.51, 5.25 };
                case CPUModel::V1:
                    return { 41.44, 5.01, 5.64 };
                case CPUModel::A510:
                    return { 7.83, 2.53, 2.71 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_interleaved_bf16fp32_mmla_8x12;
    cls_a64_interleaved_bf16fp32_mmla_8x12(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A510:
                kernel=kai_ops_a64_interleaved_bf16fp32_mmla_8x12_a510;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
