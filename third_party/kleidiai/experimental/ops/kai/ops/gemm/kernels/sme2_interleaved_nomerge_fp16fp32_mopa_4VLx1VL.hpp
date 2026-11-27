//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include "common_internal/kernel_args.hpp"
#include "../std_transforms_sme.hpp"

namespace kai {
namespace ops {

// Implementations
extern "C" void kai_ops_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL(const KernelArgs *);

class cls_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL
{
public:
    typedef __fp16 lhs_operand_type;
    typedef __fp16 rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const KernelArgs *);

    /* Kernel blocking parameters */
    static unsigned int out_height()
    {
        return sme::get_vector_length<float>() * 4;
    }

    static unsigned int out_width()
    {
        return sme::get_vector_length<float>() * 1;
    }

    static constexpr unsigned int k_unroll()
    {
        return 2;
    }

    static constexpr bool supports_bias()
    {
        return true;
    }

    static constexpr bool is_sme()
    {
        return true;
    }

    static constexpr size_t accumulation_buffer_element_size()
    {
        return sizeof(float);
    }

    // Default to the generic kernel
    kern_type kernel = kai_ops_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL;

    StdTransformsSME<lhs_operand_type, result_type, 4, 1, 2> transforms = {};

    cls_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai
