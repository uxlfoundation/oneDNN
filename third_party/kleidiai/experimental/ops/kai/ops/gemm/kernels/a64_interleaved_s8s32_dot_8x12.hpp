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
extern "C" void kai_ops_a64_interleaved_s8s32_dot_8x12( const KernelArgs * );
extern "C" void kai_ops_a64_interleaved_s8s32_dot_8x12_a55( const KernelArgs * );
extern "C" void kai_ops_a64_interleaved_s8s32_dot_8x12_x1( const KernelArgs * );

class cls_a64_interleaved_s8s32_dot_8x12
{
public:
    typedef int8_t lhs_operand_type;
    typedef int8_t rhs_operand_type;
    typedef int32_t result_type;

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

        if (std::is_same<T, int32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 31.82, 3.51, 8.03 };
                case CPUModel::A510:
                    return { 19.73, 3.38, 3.70 };
                case CPUModel::V1:
                    return { 61.58, 4.78, 10.83 };
                case CPUModel::A55r1:
                    return { 14.286, 1.171, 1.209 };
            }
        }


        if (std::is_same<T, int8_t>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 15.361, 0.9341, 0.1636 };
                default:
                    return { 29.0698, 3.9793, 0.4003 };
                case CPUModel::A510:
                    return { 19.73, 3.38, 0.27 };
                case CPUModel::V1:
                    return { 51.14, 7.38, 0.65 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=kai_ops_a64_interleaved_s8s32_dot_8x12;
    cls_a64_interleaved_s8s32_dot_8x12(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
                kernel=kai_ops_a64_interleaved_s8s32_dot_8x12_a55;
                break;
            case CPUModel::X1:
                kernel=kai_ops_a64_interleaved_s8s32_dot_8x12_x1;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
