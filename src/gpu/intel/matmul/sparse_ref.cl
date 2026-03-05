/*******************************************************************************
* Copyright 2024 Intel Corporation
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
#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"

__kernel void ref_sparse_matmul(__global const SRC_DATA_T *A_values,
        __global const int *A_rows, __global const int *A_cols,
        __global const WEI_DATA_T *B, __global DST_DATA_T *C, const dim_t nnz) {

    size_t m = get_global_id(0);
    size_t n = get_global_id(1);

    // initialize dense destination tensor
    dim_t dst_off = DST_OFF(m, n, 0, 0, 0);
    ACC_DATA_T accum = 0;

    for (dim_t idx = 0; idx < nnz; idx++) {
        int a_row = A_rows[idx];
        if (a_row == m) {
            int a_col = A_cols[idx];
            ACC_DATA_T val = CONCAT2(into_, ACC_DATA_T)(A_values[idx]);
            dim_t wei_off = WEI_OFF(0, a_col, n, 0, 0, 0);
            accum += val * CONCAT2(into_, ACC_DATA_T)(B[wei_off]);
        }
    }

    write(C + dst_off, accum);
}
