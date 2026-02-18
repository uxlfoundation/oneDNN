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

// Fills a buffer with pseudo-random uint32 values using lowbias32 hash
// (Chris Wellons). Each work-item produces one uint32 from its global ID
// XORed with a caller-provided seed.The mask 0xEEEEEEEE clears bit 0 and bit 4
// of every byte, which breaks all-ones exponent for every supported FP format.
__kernel void fill_random(__global uint *buf, uint seed, uint count) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uint h = gid ^ seed;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    h &= 0xEEEEEEEEu;
    buf[gid] = h;
}
