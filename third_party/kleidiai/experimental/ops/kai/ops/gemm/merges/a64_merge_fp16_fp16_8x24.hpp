//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

extern "C" void kai_ops_a64_merge_fp16_fp16_8x24(const KernelArgs *ka);
template<>
void MergeResults<24, 8, false, __fp16, __fp16>(
    const KernelArgs *ka)
{
    kai_ops_a64_merge_fp16_fp16_8x24(ka);
}
