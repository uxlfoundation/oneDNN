/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "gpu/intel/include/io.h"

#if DT_F64
typedef double8 acc8_t;
#else
typedef float8 acc8_t;
#endif

acc8_t get_values(__global SRC_DATA_T *src, ptrdiff_t offset, dim_t nelems) {
    acc8_t val = 0;
    const uint max_sub_group_size = get_max_sub_group_size();
    if (offset + VECT_DT_N * max_sub_group_size < nelems) {
        val = block_load(val, src + offset);
    } else {
        const ptrdiff_t sub_group_local_id = get_sub_group_local_id();
        ptrdiff_t pos = offset + sub_group_local_id;
        for (uint i = 0; pos < nelems && i < VECT_DT_N; i++) {
            val[i] = load(val[i], src, pos);
            pos += max_sub_group_size;
        }
    }
    return val;
}

__kernel void xe_sum(__global SRC_DATA_T *input0, __global SRC_DATA_T *input1,
        __global SRC_DATA_T *input2, __global SRC_DATA_T *input3,
        __global SRC_DATA_T *input4, __global SRC_DATA_T *input5,
        __global SRC_DATA_T *input6, __global SRC_DATA_T *input7,
        __global SRC_DATA_T *input8, __global SRC_DATA_T *input9,
        __global SRC_DATA_T *input10, __global SRC_DATA_T *input11,
        __global SRC_DATA_T *input12, __global SRC_DATA_T *input13,
        __global SRC_DATA_T *input14, __global SRC_DATA_T *input15,
        __global DST_DATA_T *output, __global float *scales, dim_t nelems) {

    const uint group_id = get_group_id(0);
    const uint group_size = get_local_size(0);
    const uint sub_group_id = get_sub_group_id();
    const uint max_sub_group_size = get_max_sub_group_size();
    const uint sub_group_local_id = get_sub_group_local_id();

    ptrdiff_t offset
            = (group_id * group_size + sub_group_id * max_sub_group_size)
            * VECT_DT_N;

    int id = 0;
    acc8_t sum = 0;
    if (id < N_INPUTS) sum += get_values(input0, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input1, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input2, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input3, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input4, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input5, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input6, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input7, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input8, offset, nelems) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input9, offset, nelems) * scales[id++];
    if (id < N_INPUTS)
        sum += get_values(input10, offset, nelems) * scales[id++];
    if (id < N_INPUTS)
        sum += get_values(input11, offset, nelems) * scales[id++];
    if (id < N_INPUTS)
        sum += get_values(input12, offset, nelems) * scales[id++];
    if (id < N_INPUTS)
        sum += get_values(input13, offset, nelems) * scales[id++];
    if (id < N_INPUTS)
        sum += get_values(input14, offset, nelems) * scales[id++];
    if (id < N_INPUTS)
        sum += get_values(input15, offset, nelems) * scales[id++];

    if (offset + VECT_DT_N * max_sub_group_size < nelems) {
        block_write(output + offset, &sum);
    } else {
        ptrdiff_t pos = offset + sub_group_local_id;
        for (uint i = 0; pos < nelems && i < VECT_DT_N; i++) {
            write(output + pos, sum[i]);
            pos += max_sub_group_size;
        }
    }
}
