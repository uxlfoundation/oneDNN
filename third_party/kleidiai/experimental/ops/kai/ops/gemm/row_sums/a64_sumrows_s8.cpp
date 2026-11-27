//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#include "common_internal/kernel_args.hpp"
#include "common_internal/quantized.hpp"

extern "C" void kai_ops_a64_sumrows_s8(const KernelArgs *ka, int32_t *out_ptr, int32_t rhs_offset);

namespace kai {
namespace ops {

template<>
void row_sums_indirect<int8_t>(const KernelArgs *ka, int32_t *out_ptr, int32_t rhs_offset)
{
    kai_ops_a64_sumrows_s8(ka, out_ptr, rhs_offset);
}

} // namespace ops
} // namespace kai

