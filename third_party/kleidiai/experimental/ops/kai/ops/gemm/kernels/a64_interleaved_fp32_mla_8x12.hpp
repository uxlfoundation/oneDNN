//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
extern "C" void kai_ops_a64_interleaved_fp32_mla_8x12( const KernelArgs * );
extern "C" void kai_ops_a64_interleaved_fp32_mla_8x12_a55( const KernelArgs * );
extern "C" void kai_ops_a64_interleaved_fp32_mla_8x12_x1( const KernelArgs * );

class cls_a64_interleaved_fp32_mla_8x12
{
public:
    typedef float lhs_operand_type;
    typedef float rhs_operand_type;
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
        return 1;
    }

    static constexpr size_t accumulation_buffer_element_size()
    {
        return 0;
    }


    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 12, 1> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 12, 1, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 3.954, 1.252, 1.141 };
                default:
                    return { 7.2307, 3.876, 2.932 };
                case CPUModel::A73:
                    return { 2.885, 1.429, 1.163 };
                case CPUModel::A53:
                    return { 2.7, 0.9, 0.8 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_interleaved_fp32_mla_8x12;
    cls_a64_interleaved_fp32_mla_8x12(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
            case CPUModel::A53:
                kernel=kai_ops_a64_interleaved_fp32_mla_8x12_a55;
                break;
            case CPUModel::X1:
                kernel=kai_ops_a64_interleaved_fp32_mla_8x12_x1;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
