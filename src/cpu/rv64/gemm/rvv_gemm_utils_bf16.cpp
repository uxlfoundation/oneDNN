/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "cpu/rv64/gemm/rvv_gemm_utils_bf16.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

#define BM_NOCOPY_RVV_BF16 64
#define BN_NOCOPY_RVV_BF16 48
#define BK_NOCOPY_RVV_BF16 384
#define BN_LARGE_NOCOPY_RVV_BF16 192
#define BM_SMALL_NOCOPY_RVV_BF16 16
#define BN_SMALL_NOCOPY_RVV_BF16 1
#define BK_SMALL_NOCOPY_RVV_BF16 4

// Threading calculation for bf16 GEMM, similar to f32 version
void calc_nthr_nocopy_rvv_bf16(dim_t m, dim_t n, dim_t k, int nthrs, int *nthrs_m,
        int *nthrs_n, int *nthrs_k, dim_t *BM, dim_t *BN, dim_t *BK) {
    
    int nthr_m = 1, nthr_n = 1, nthr_k = 1;
    dim_t bm = BM_NOCOPY_RVV_BF16, bn = BN_NOCOPY_RVV_BF16, bk = BK_NOCOPY_RVV_BF16;

    if (nthrs <= 1) {
        *nthrs_m = nthr_m;
        *nthrs_n = nthr_n;
        *nthrs_k = nthr_k;
        *BM = bm;
        *BN = bn;
        *BK = bk;
        return;
    }

    // For small problems, use smaller block sizes
    if (m * n * k < 1000000) {
        bm = BM_SMALL_NOCOPY_RVV_BF16;
        bn = BN_SMALL_NOCOPY_RVV_BF16;
        bk = BK_SMALL_NOCOPY_RVV_BF16;
    }

    // Simple heuristic for thread distribution
    if (m >= n && m >= k) {
        nthr_m = nthrs;
    } else if (n >= k) {
        nthr_n = nthrs;
    } else {
        nthr_k = nthrs;
    }

    // Adjust for large N dimension
    if (n > 1000) {
        bn = BN_LARGE_NOCOPY_RVV_BF16;
    }

    *nthrs_m = nthr_m;
    *nthrs_n = nthr_n;
    *nthrs_k = nthr_k;
    *BM = bm;
    *BN = bn;
    *BK = bk;
}

void partition_unit_diff_bf16(
        int ithr, int nthr, dim_t n, dim_t *t_offset, dim_t *t_block) {
    dim_t band = n / nthr;
    dim_t tail = n % nthr;
    if (ithr < tail) {
        band++;
        *t_offset = band * ithr;
    } else {
        *t_offset = band * ithr + tail;
    }
    *t_block = band;
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl