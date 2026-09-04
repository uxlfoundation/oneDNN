//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once
#include "../performance_parameters.hpp"
#include "../std_transforms_sve.hpp"
#include "kai/ops/bfloat.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<bfloat16>, \
    size_t, size_t, \
    const bfloat16 *, \
    OutputArg<float>, \
    const float *, Activation, bool

namespace kai {
namespace ops {

// Actual kernel implementations
void sve_hybrid_bf16fp32_dot_6x4VL( ARGLIST );

class cls_sve_hybrid_bf16fp32_dot_6x4VL
{
public:
    typedef bfloat16 lhs_operand_type;
    typedef bfloat16 rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }

    static unsigned int out_width()
    {
        return get_vector_length<float>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 2;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 4, 2> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, bfloat16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 15.83 };
                case CPUModel::A510:
                    return { 6.80 };
                case CPUModel::V1:
                    return { 31.55 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_hybrid_bf16fp32_dot_6x4VL;
    cls_sve_hybrid_bf16fp32_dot_6x4VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai

#undef ARGLIST
