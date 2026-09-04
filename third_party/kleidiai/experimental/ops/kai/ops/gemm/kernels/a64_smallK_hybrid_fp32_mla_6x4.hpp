//
// SPDX-FileCopyrightText: Copyright 2019-2020, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __aarch64__



namespace kai {
namespace ops {

// Actual kernel implementations
void a64_smallK_hybrid_fp32_mla_6x4(const float *, int, const float *, float *, int, int, int, int, const float *, Activation, bool);

class cls_a64_smallK_hybrid_fp32_mla_6x4
{
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, int, const float *, float *, int, int, int, int, const float *, Activation, bool);

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }

    static unsigned int out_width()
    {
        return 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    static constexpr bool supports_bias()
    {
        return true;
    }

    static constexpr bool supports_activation()
    {
        return true;
    }

    StdTransformsFixed<operand_type, operand_type, result_type, 6, 4, 1> transforms = {};

    // Default to the generic kernel
    kern_type kernel=a64_smallK_hybrid_fp32_mla_6x4;

    cls_a64_smallK_hybrid_fp32_mla_6x4(const CPUInfo *)
    {

    }
};

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
