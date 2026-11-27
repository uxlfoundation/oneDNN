//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

extern "C" void kai_ops_sve_merge_s32_s32_8x3VL(const KernelArgs *ka);
template<>
void MergeResults<3, 8, true, int32_t, int32_t>(
    const KernelArgs *ka)
{
    kai_ops_sve_merge_s32_s32_8x3VL(ka);
}
