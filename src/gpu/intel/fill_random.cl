/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/intel/include/philox.h"

// Fills a buffer with pseudo-random data using Philox RNG and subgroup block
// writes. Each subgroup (16 work-items) writes 256 bytes.
#define SG_SIZE 16
#define BYTES_PER_SG (SG_SIZE * 4 * 4)

__attribute__((intel_reqd_sub_group_size(SG_SIZE))) __kernel void fill_random(
        __global uchar *buf, uint seed, ulong byte_count) {
    const ulong base = (get_global_id(0) / SG_SIZE) * BYTES_PER_SG;
    if (base >= byte_count) return;

    const uint b = (uint)get_global_id(0) * 4;
    uchar16 rnd
            = as_uchar16(philox_4x32_vec4(b, b ^ seed) & (uint4)(0xEEEEEEEEu));

    if (base + BYTES_PER_SG <= byte_count) {
        intel_sub_group_block_write_uc16(buf + base, rnd);
        return;
    }

    const uint lid = get_sub_group_local_id();
    unroll_for(int i = 0; i < 16; i++) {
        ulong off = base + lid + (ulong)i * SG_SIZE;
        if (off < byte_count) buf[off] = rnd[i];
    }
}
