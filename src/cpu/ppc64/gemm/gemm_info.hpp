/*******************************************************************************
* Copyright 2022 IBM Corporation
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

#ifndef CPU_PPC64_GEMM_GEMM_INFO_HPP
#define CPU_PPC64_GEMM_GEMM_INFO_HPP

#include <cstdint>
#include <memory>

#include "common/c_types_map.hpp"
#include "cpu/ppc64/gemm/gemm_pack_storage.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace ppc64 {

enum class pack_type { none, pack_a, pack_b };

enum class offset_type {
    none,
    fixed,
    column,
    row,
};

// Indices for kernel arrays. TODO Is it okay to place this here?
enum { no_sum = 0, do_sum = 1 };
enum { no_trans = 0, do_trans = 1, packed = 2 };
enum { no_beta0 = 0, do_beta0 = 1 };
enum { no_alpha1 = 0, do_alpha1 = 1 };

template <typename a_t, typename b_t, typename c_t>
struct gemm_info_t {

    // Interface arguments.
    int transa, transb;
    offset_type offsetc;
    dim_t m, n, k;
    dim_t lda, ldb, ldc;
    const a_t *a;
    const b_t *b;
    c_t *c;
    float alpha, beta;

    bool b_is_signed;
    int32_t ao;
    int32_t bo;
    const c_t *co;

    pack_type packing;
    gemm_pack_storage_t *pack_dst;
    bool measure_only;
    std::shared_ptr<const gemm_pack_storage_t> a_packed, b_packed;

    // Kernel parameters.
    dim_t um, un, uk, bm, bn, bk;
    dim_t bn_small_k, bk_traditional, blocking_small_k;

    // Gemv parameters
    int swap = false;
    gemm_info_t(const char *transA, const char *transB, const char *offsetC,
            const dim_t *m, const dim_t *n, const dim_t *k, const float *alpha,
            const a_t *a, const dim_t *lda, const a_t *oa, const b_t *b,
            const dim_t *ldb, const b_t *ob, const float *beta, c_t *c,
            const dim_t *ldc, const c_t *oc, bool force_nocopy,
            pack_type packing, gemm_pack_storage_t *pack_dst,
            bool measure_only);

    void update_blocking(const gemm_threading_t &thread_info);
};
} // namespace ppc64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_PPC64_GEMM_GEMM_INFO_HPP
