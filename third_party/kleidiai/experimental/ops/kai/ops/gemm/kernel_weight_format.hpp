//
// SPDX-FileCopyrightText: Copyright 2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kai/ops/gemm/kai_ops.hpp"

namespace kai {
namespace ops {

/* Internal enum to define the weight format a kernel is expecting.
 *
 * This is distinct from the "external" WeightFormat defined in kai_ops.hpp primarily to allow for SVE, where
 * internally kernels are defined in terms of multiples of the SVE vector length, but externally they are converted
 * to a fixed format (based on the VL of the machine we are running on).
 *
 * Encoded as a bitfield:
 *  bit     0 : SVE flag
 *  bit     4 : BF16 convert flag (fast mode)
 *  bits 11-8 : block length (bytes)
 *  bits 15-12: vector count
 */
enum class KernelWeightFormat {
    NON_FIXED        = 0,
    VL128_BL16       = 0x1200,
    VL128_BL32       = 0x1400,
    VL128_BL32_BF16  = 0x1410,
    VL128_BL64       = 0x1800,
    VL256_BL64       = 0x2800,
    VL256_BL64_BF16  = 0x2810,
    VL1VL_BL16       = 0x1201,
    VL1VL_BL32       = 0x1401,
    VL1VL_BL32_BF16  = 0x1411,
    VL1VL_BL64       = 0x1801,
    VL2VL_BL64       = 0x2801,
    VL2VL_BL64_BF16  = 0x2811
};

WeightFormat get_weight_format(const KernelWeightFormat, size_t);

}  // namespace ops
}  // namespace kai
