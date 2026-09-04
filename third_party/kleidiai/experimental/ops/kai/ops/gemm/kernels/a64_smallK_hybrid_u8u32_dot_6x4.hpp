//
// SPDX-FileCopyrightText: Copyright 2019-2020, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __aarch64__

#include <cstdint>

namespace kai {
namespace ops {

// Actual kernel implementations
void a64_smallK_hybrid_u8u32_dot_6x4(const uint8_t *, int, const uint8_t *, uint32_t *, int, int, int, int, const uint32_t *, Activation, bool);
void a64_smallK_hybrid_u8u32_dot_6x4_a55(const uint8_t *, int, const uint8_t *, uint32_t *, int, int, int, int, const uint32_t *, Activation, bool);

class cls_a64_smallK_hybrid_u8u32_dot_6x4
{
public:
    typedef uint8_t operand_type;
    typedef uint32_t result_type;

    typedef void (*kern_type)(const uint8_t *, int, const uint8_t *, uint32_t *, int, int, int, int, const uint32_t *, Activation, bool);

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
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    static constexpr bool supports_bias()
    {
        return false;
    }

    static constexpr bool supports_activation()
    {
        return false;
    }

    StdTransformsFixed<operand_type, operand_type, result_type, 6, 4, 4> transforms = {};

    // Default to the generic kernel
    kern_type kernel=a64_smallK_hybrid_u8u32_dot_6x4;

    cls_a64_smallK_hybrid_u8u32_dot_6x4(const CPUInfo *ci)
    {
        if (ci->get_cpu_model() == CPUModel::A55r1) {
            kernel = a64_smallK_hybrid_u8u32_dot_6x4_a55;
        }
    }
};

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
