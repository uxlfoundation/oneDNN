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

inline uint4 fill_random_philox4x32(uint idxLo, uint idxHi, uint seedLo, uint seedHi)
{
    uint4 ctr = (uint4)(idxLo, idxHi, seedLo, seedHi);
    uint key0 = seedLo, key1 = seedHi;

    const uint M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    const uint W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;

#pragma unroll
    for (int round = 0; round < 10; round++) {
        ulong p0 = (ulong)ctr.s0 * (ulong)M0;
        ulong p1 = (ulong)ctr.s2 * (ulong)M1;
        uint hi0 = (uint)(p0 >> 32), lo0 = (uint)p0;
        uint hi1 = (uint)(p1 >> 32), lo1 = (uint)p1;

        ctr = (uint4)(hi1 ^ ctr.s1 ^ key0, lo1, hi0 ^ ctr.s3 ^ key1, lo0);

        key0 += W0;
        key1 += W1;
    }

    return ctr;
}

__kernel void fill_random_kernel(
        __global uint *buf, uint seed, ulong byte_count, uint mask)
{
    const ulong item = get_global_id(0);
    const ulong start = item * 16;
    if (start >= byte_count) return;

    uint idx = (uint)item;
    uint4 rnd = fill_random_philox4x32(idx, idx ^ 0x9E3779B9u, seed, seed ^ 0x1F123BB5u) & (uint4)(mask);

    if (start + 16 <= byte_count) {
        vstore4(rnd, item, buf);
    } else {
        __global uchar *p = (__global uchar *)buf;
        uchar16 bytes = as_uchar16(rnd);
        for (ulong i = 0; i < byte_count - start; i++)
            p[start + i] = bytes[i];
    }
}
