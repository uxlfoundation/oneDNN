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
extern "C" void kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL(const KernelArgs *);

class cls_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL
{
public:
    typedef int8_t lhs_operand_type;
    typedef int8_t rhs_operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const KernelArgs *);

    /* Kernel blocking parameters */
    static unsigned int out_height()
    {
        return sme::get_vector_length<int32_t>() * 1;
    }

    static unsigned int out_width()
    {
        return sme::get_vector_length<int32_t>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
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
        return sizeof(int32_t);
    }

    // Default to the generic kernel
    kern_type kernel = kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL;

    StdTransformsSME<lhs_operand_type, result_type, 1, 4, 4> transforms = {};

    cls_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL(const CPUInfo *)
    {
    }
};

} // namespace ops
} // namespace kai
