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
// (Chris Wellons). Each thread produces BLOCK_SIZE uint32 values using SIMD.
__kernel void fill_random(__global uint *buf) {
    uint i, start_id = (uint)(get_global_id(0) * BLOCK_SIZE);

    DT idx = (DT)(0);
#pragma unroll SIMD_WIDTH
    for (i = 0; i < SIMD_WIDTH; i++) {
        idx[i] = start_id + i;
    }

#pragma unroll ITERS
    for (i = 0; i < ITERS; ++i) {
        DT v = (idx ^ (DT)SEED);
        v = (v ^ (v >> 16)) * (DT)0x7FEB352Du;
        v = (v ^ (v >> 15)) * (DT)0x846CA68Bu;
        v = (v ^ (v >> 16)) & (DT)0xEEEEEEEEu;
        SIMD_STORE(v, 0, buf + start_id + (i * SIMD_WIDTH));
        idx += SIMD_WIDTH;
    }
}
