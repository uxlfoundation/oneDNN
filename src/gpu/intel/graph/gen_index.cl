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

#ifdef USE_INT32_OFFSET
typedef int idx_t;
#else
typedef long idx_t;
#endif

__kernel void gen_index(__global int *dst, int axis) {
    idx_t id = (idx_t)get_global_id(0);
    int result = 0;
    idx_t offset = 0;
    int idx;

#define PROCESS_DIM(N) \
    idx = (int)(id % D##N); \
    id = id / D##N; \
    offset += (idx_t)idx * S##N; \
    if (axis == N) result = idx;

    PROCESS_DIM(0);
#if NDIMS > 1
    PROCESS_DIM(1);
#endif
#if NDIMS > 2
    PROCESS_DIM(2);
#endif
#if NDIMS > 3
    PROCESS_DIM(3);
#endif
#if NDIMS > 4
    PROCESS_DIM(4);
#endif
#if NDIMS > 5
    PROCESS_DIM(5);
#endif
#undef PROCESS_DIM

    dst[offset] = result;
}
