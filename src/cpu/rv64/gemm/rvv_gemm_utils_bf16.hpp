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

#ifndef CPU_RV64_GEMM_RVV_GEMM_UTILS_BF16_HPP
#define CPU_RV64_GEMM_RVV_GEMM_UTILS_BF16_HPP

#include <cstddef>
#include "common/c_types_map.hpp"
#include "common/bfloat16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

// GEMM traits for bfloat16, similar to f32 version
template <typename T, bool isTransA, bool isTransB>
struct gemm_traits_t {};

template <bool isTransA, bool isTransB>
struct gemm_traits_t<bfloat16_t, isTransA, isTransB> {
    static constexpr dim_t m = 16;
    static constexpr dim_t n = 4;
    static constexpr dim_t BM = 4032;
    static constexpr dim_t BN = isTransA ? 96 : 48;
    static constexpr dim_t BK = isTransB ? 96 : 256;
};

template <typename T>
using unroll_factor_bf16 = gemm_traits_t<T, false, false>;

// Threading and blocking utilities for bf16
void calc_nthr_nocopy_rvv_bf16(dim_t m, dim_t n, dim_t k, int nthrs, int *nthrs_m,
        int *nthrs_n, int *nthrs_k, dim_t *BM, dim_t *BN, dim_t *BK);

void partition_unit_diff_bf16(
        int ithr, int nthr, dim_t n, dim_t *t_offset, dim_t *t_block);

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_GEMM_RVV_GEMM_UTILS_BF16_HPP