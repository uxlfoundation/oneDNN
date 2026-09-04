//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NO_MULTI_THREADING
#include <mutex>
#endif
#include <cstdint>

#include "kai/ops/gemm/kai_ops.hpp"
#include "kernel_weight_format.hpp"
#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

#ifndef NO_MULTI_THREADING
std::mutex report_mutex;
#endif

WeightFormat get_weight_format(const KernelWeightFormat kwf, size_t element_size) {
    if (kwf==KernelWeightFormat::NON_FIXED) {
        return WeightFormat::UNSPECIFIED;
    }

    uint32_t kwf_i = static_cast<uint32_t>(kwf);
    uint32_t wf_i = 0;

    const auto block_bytes = (kwf_i >> 8) & 0xf;
    const auto vector_count = (kwf_i >> 12) & 0xf;

    uint32_t vector_bytes;

    // For fast mode BF16 kernels set the appropriate bit and override element size to 2.
    if (kwf_i & 0x10) {
        element_size = 2;
        wf_i |= 0x10;
    }

    // Get total bytes in vector output.  Populate with NEON default, then
    // override with SVE if it is an SVE format (AArch64 only).
    vector_bytes = vector_count * 16;

#ifdef __aarch64__
    if (kwf_i & 0x1) {
        vector_bytes = vector_count * get_vector_length<uint8_t>();
    }
#endif

    auto input_blocking = block_bytes / element_size;
    auto output_blocking = vector_bytes / block_bytes;

    wf_i |= (input_blocking << 20);
    wf_i |= (output_blocking << 8);

    return static_cast<WeightFormat>(wf_i);
}

}  // namespace ops
}  // namespace kai
