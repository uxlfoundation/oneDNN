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

#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"

__kernel void ref_sparse_matmul(__global const void *sparse_values,
        __global const int *sparse_meta0, __global const int *sparse_meta1,
        __global const void *dense, __global DST_DATA_T *C, const dim_t nnz) {

    size_t m = get_global_id(0);
    size_t n = get_global_id(1);

    dim_t dst_off = DST_OFF(m, n, 0, 0, 0);
    float accum = 0.0f;

#if SPARSE_WEI
    // Weights (B) are sparse, src (A) is dense.
    // C[m,n] = sum_k A[m,k] * B[k,n]
    __global const WEI_DATA_T *B_values
            = (__global const WEI_DATA_T *)sparse_values;
    __global const SRC_DATA_T *A = (__global const SRC_DATA_T *)dense;
#if IS_CSR
    // CSR on B: sparse_meta1 = row_ptrs, sparse_meta0 = col_indices
    // Iterate over rows k; for each nnz in row k with col == n, accumulate.
    for (dim_t k = 0; k < K; k++) {
        int row_start = sparse_meta1[k];
        int row_end = sparse_meta1[k + 1];
        for (int idx = row_start; idx < row_end; idx++) {
            if (sparse_meta0[idx] == n) {
                dim_t src_off = SRC_OFF(m, k, 0, 0, 0);
                accum += SRC_TO_REF(A[src_off]) * WEI_TO_REF(B_values[idx]);
            }
        }
    }
#else
    // COO on B: sparse_meta0 = rows, sparse_meta1 = cols
    for (dim_t idx = 0; idx < nnz; idx++) {
        int b_row = sparse_meta0[idx];
        int b_col = sparse_meta1[idx];
        if (b_col == n) {
            dim_t src_off = SRC_OFF(m, b_row, 0, 0, 0);
            accum += SRC_TO_REF(A[src_off]) * WEI_TO_REF(B_values[idx]);
        }
    }
#endif
#else
    // Src (A) is sparse, weights (B) are dense.
    // C[m,n] = sum_k A[m,k] * B[k,n]
    __global const SRC_DATA_T *A_values
            = (__global const SRC_DATA_T *)sparse_values;
    __global const WEI_DATA_T *B = (__global const WEI_DATA_T *)dense;
#if IS_CSR
    // CSR on A: sparse_meta1 = row_ptrs, sparse_meta0 = col_indices
    int row_start = sparse_meta1[m];
    int row_end = sparse_meta1[m + 1];
    for (int idx = row_start; idx < row_end; idx++) {
        int a_col = sparse_meta0[idx];
        dim_t wei_off = WEI_OFF(0, a_col, n, 0, 0, 0);
        accum += SRC_TO_REF(A_values[idx]) * WEI_TO_REF(B[wei_off]);
    }
#else
    // COO on A: sparse_meta0 = rows, sparse_meta1 = cols
    for (dim_t idx = 0; idx < nnz; idx++) {
        int a_row = sparse_meta0[idx];
        if (a_row == m) {
            int a_col = sparse_meta1[idx];
            dim_t wei_off = WEI_OFF(0, a_col, n, 0, 0, 0);
            accum += SRC_TO_REF(A_values[idx]) * WEI_TO_REF(B[wei_off]);
        }
    }
#endif
#endif

    C[dst_off] = TO_DST(accum);
}
