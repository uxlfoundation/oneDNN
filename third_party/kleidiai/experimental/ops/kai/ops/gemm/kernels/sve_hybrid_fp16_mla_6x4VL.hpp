//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../performance_parameters.hpp"
#include "../std_transforms_sve.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<__fp16>, \
    size_t, size_t, \
    const __fp16 *, \
    OutputArg<__fp16>, \
    const __fp16 *, Activation, bool

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_hybrid_fp16_mla_6x4VL( ARGLIST );
void sve_hybrid_fp16_mla_6x4VL_a64fx( ARGLIST );

class cls_sve_hybrid_fp16_mla_6x4VL
{
public:
    typedef __fp16 lhs_operand_type;
    typedef __fp16 rhs_operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }

    static unsigned int out_width()
    {
        return get_vector_length<__fp16>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 4, 1> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, __fp16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 15.90 };
                case CPUModel::A510:
                    return { 12.44 };
                case CPUModel::V1:
                    return { 31.51 };
                case CPUModel::A64FX:
                    return { 49.14 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_hybrid_fp16_mla_6x4VL;
    cls_sve_hybrid_fp16_mla_6x4VL(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A64FX:
                kernel=sve_hybrid_fp16_mla_6x4VL_a64fx;
                break;
        }
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
