//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

extern "C" void kai_ops_a64_merge_u32_u32_8x12(const KernelArgs *ka);
template<>
void MergeResults<12, 8, false, uint32_t, uint32_t>(
    const KernelArgs *ka)
{
    kai_ops_a64_merge_u32_u32_8x12(ka);
}
