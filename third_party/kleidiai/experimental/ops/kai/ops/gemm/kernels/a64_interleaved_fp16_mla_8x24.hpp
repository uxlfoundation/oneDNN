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
extern "C" void kai_ops_a64_interleaved_fp16_mla_8x24( const KernelArgs * );
extern "C" void kai_ops_a64_interleaved_fp16_mla_8x24_a55( const KernelArgs * );
extern "C" void kai_ops_a64_interleaved_fp16_mla_8x24_x1( const KernelArgs * );

class cls_a64_interleaved_fp16_mla_8x24
{
public:
    typedef __fp16 lhs_operand_type;
    typedef __fp16 rhs_operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( const KernelArgs * );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return 24;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr size_t accumulation_buffer_element_size()
    {
        return 0;
    }


    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 24, 24, 1> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 24, 24, 1, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, __fp16>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 7.16, 1.14, 0.67 };
                default:
                    return { 12.67, 3.98, 1.16 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_interleaved_fp16_mla_8x24;
    cls_a64_interleaved_fp16_mla_8x24(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
                kernel=kai_ops_a64_interleaved_fp16_mla_8x24_a55;
                break;
            case CPUModel::X1:
                kernel=kai_ops_a64_interleaved_fp16_mla_8x24_x1;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
