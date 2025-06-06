/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#define INF_NAN_MASK 0xBFFFBFFF
#include "gpu/intel/ocl/ocl_philox.h"

__kernel void ocl_philox_kernel(__global uint *data, ulong nbytes, ulong seed) {
    const ulong gid = get_global_id(0);
    const ulong block_size = 4;
    const ulong offset = gid * block_size;
    const ulong working_items = nbytes >> 2;

    if (offset >= working_items) return;

    uint4 rands = ref_philox_4x32(offset, seed) & (uint4)(INF_NAN_MASK);
    const ulong tail = working_items - offset;

    if (tail >= block_size) {
        vstore4(rands, 0, data + offset);
    } else {
        if (tail >= 1) data[offset] = rands.s0;
        if (tail >= 2) data[offset + 1] = rands.s1;
        if (tail == 3) data[offset + 2] = rands.s2;
    }
}
