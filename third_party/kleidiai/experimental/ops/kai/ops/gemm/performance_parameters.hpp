//
// SPDX-FileCopyrightText: Copyright 2020, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace kai {
namespace ops {

struct PerformanceParameters {
    float  kernel_macs_cycle;
    float  prepare_bytes_cycle = 0.0f;
    float  merge_bytes_cycle   = 0.0f;

    PerformanceParameters(float k) : kernel_macs_cycle(k) { }
    PerformanceParameters(float k, float p, float m) : kernel_macs_cycle(k), prepare_bytes_cycle(p), merge_bytes_cycle(m) { }
};

}  // namespace ops
}  // namespace kai
