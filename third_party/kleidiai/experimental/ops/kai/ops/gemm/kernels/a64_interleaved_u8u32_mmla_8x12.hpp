//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
extern "C" void kai_ops_a64_interleaved_u8u32_mmla_8x12( const KernelArgs * );
extern "C" void kai_ops_a64_interleaved_u8u32_mmla_8x12_a510( const KernelArgs * );

class cls_a64_interleaved_u8u32_mmla_8x12
{
public:
    typedef uint8_t lhs_operand_type;
    typedef uint8_t rhs_operand_type;
    typedef uint32_t result_type;

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
        return 8;
    }

    static constexpr size_t accumulation_buffer_element_size()
    {
        return 0;
    }


    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 12, 8> transforms = {};
    StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 12, 8, true> transforms_quantized = {};
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
    kern_type kernel=kai_ops_a64_interleaved_u8u32_mmla_8x12;
    cls_a64_interleaved_u8u32_mmla_8x12(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A510:
                kernel=kai_ops_a64_interleaved_u8u32_mmla_8x12_a510;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
