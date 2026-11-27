//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

extern "C" void kai_ops_a64_merge_fp32_bf16_8x12(const KernelArgs *ka);
template<>
void MergeResults<12, 8, false, float, bfloat16>(
    const KernelArgs *ka)
{
    kai_ops_a64_merge_fp32_bf16_8x12(ka);
}
