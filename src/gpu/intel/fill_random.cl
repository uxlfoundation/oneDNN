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

#define DT_UNDEF 1
#include "gpu/intel/include/philox.h"

// Fills a buffer with pseudo-random uint32 values using the Philox PRNG.
__kernel void fill_random(__global uint *buf, uint seed, uint byte_count) {
    uint gid = (uint)get_global_id(0);
    uint offset = gid * 4;
    if (offset >= byte_count) return;

    uint value = philox_4x32(gid, gid ^ seed) & 0xEEEEEEEEu;
    uint tail = byte_count - offset;

    if (tail >= 4) {
        buf[gid] = value;
    } else {
        __global uchar *p = (__global uchar *)(buf + gid);
        for (uint i = 0; i < tail; i++)
            p[i] = (uchar)(value >> (i * 8));
    }
}
