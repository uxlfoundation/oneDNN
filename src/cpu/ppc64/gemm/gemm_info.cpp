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

#include <cstdint>
#include <memory>
#include <mutex>

#include "common/dnnl_traits.hpp"
#include "cpu/gemm/gemm.hpp"
#include "cpu/ppc64/gemm/gemm_info.hpp"
#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace ppc64 {

static inline int decode_trans(char trans) {
    switch (trans) {
        case 'T':
        case 't': return do_trans;
        case 'P':
        case 'p': return packed;
        default: return no_trans;
    }
}

namespace {
template <typename b_t> // XXX for float and bfloat
void prepare_bo(int32_t &bo_gemm_info, const b_t *bo_orig) {
    UNUSED(bo_orig);
    bo_gemm_info = 0;
}
template <>
void prepare_bo(int32_t &bo_gemm_info, const uint8_t *bo_orig) {
    bo_gemm_info = bo_orig ? *bo_orig : 0;
}
template <>
void prepare_bo(int32_t &bo_gemm_info, const int8_t *bo_orig) {
    int bo_s32 = bo_orig ? *bo_orig : 0;
    bo_s32 += 128;
    bo_gemm_info = bo_s32;
}

} // namespace

template <typename a_t, typename b_t, typename c_t>
gemm_info_t<a_t, b_t, c_t>::gemm_info_t(const char *transA, const char *transB,
        const char *offsetC, const dim_t *m, const dim_t *n, const dim_t *k,
        const float *alpha, const a_t *a, const dim_t *lda, const a_t *oa,
        const b_t *b, const dim_t *ldb, const b_t *ob, const float *beta,
        c_t *c, const dim_t *ldc, const c_t *oc, bool force_nocopy,
        pack_type packing, gemm_pack_storage_t *pack_dst, bool measure_only) {

    this->transa = decode_trans(*transA);
    this->transb = decode_trans(*transB);

    this->m = *m;
    this->n = *n;
    this->k = *k;

    this->a = a;
    this->b = b;
    this->c = c;

    this->lda = lda ? *lda : 0;
    this->ldb = ldb ? *ldb : 0;
    this->ldc = ldc ? *ldc : 0;

    this->ao = 0;
    this->bo = 0;
    this->co = nullptr;

    this->alpha = alpha ? *alpha : 1.0f;
    this->beta = beta ? *beta : 1.0f;

    this->offsetc = offset_type::none;

    this->packing = packing;
    this->pack_dst = pack_dst;
    this->measure_only
            = measure_only && pack_dst && (packing != pack_type::none);

    if (this->transa == packed) {
        dim_t cols;

        this->a_packed.reset(new gemm_pack_storage_t(a));
        if (this->a_packed->get_nocopy(this->transa, this->lda, cols)) {
            this->a = this->a_packed->template matrix<a_t>();
            this->a_packed = nullptr;
        }
    }

    if (this->transb == packed) {
        dim_t rows;

        this->b_packed.reset(new gemm_pack_storage_t(b));
        if (this->b_packed->get_nocopy(this->transb, this->ldb, rows)) {
            this->b = this->b_packed->template matrix<b_t>();
            this->b_packed = nullptr;
        }
    }

    constexpr bool is_int8 = utils::one_of(
            data_traits_t<a_t>::data_type, data_type::s8, data_type::u8);
    if (is_int8) this->ao = oa ? *oa : a_t(0);
    prepare_bo<b_t>(this->bo, ob);

    this->b_is_signed = false;

    if (data_traits_t<b_t>::data_type == data_type::s8)
        this->b_is_signed = true;

    if (offsetC != nullptr) {
        char offsetc = *offsetC;
        if (offsetc == 'F' || offsetc == 'f') {
            this->offsetc = offset_type::fixed;
        } else if (offsetc == 'R' || offsetc == 'r') {
            this->offsetc = offset_type::row;
        } else { // offsetc == 'C' || offsetc == 'c'
            this->offsetc = offset_type::column;
        }
        this->co = oc;
    }

    // Blocking of M, N and K
    this->um = 16;
    this->un = 4;
    this->uk = 1;
    this->bm = 4096;
    this->bn = 128;
    this->bk = 128;
    this->bk_traditional = 128;
    this->blocking_small_k = 64;
    this->bn_small_k = 16;
}

template <typename a_t, typename b_t, typename c_t>
void gemm_info_t<a_t, b_t, c_t>::update_blocking(
        const gemm_threading_t &thread_info) {

    if (thread_info.block_m > 0) this->bm = thread_info.block_m;
    if (thread_info.block_n > 0) this->bn = thread_info.block_n;
    if (thread_info.block_k > 0) this->bk = thread_info.block_k;
}

// Instantiate the gemm_info_t templates needed.
template // For gemm_s8u8s32
        struct gemm_info_t<int8_t, uint8_t, int32_t>;

template // For gemm_s8s8s32
        struct gemm_info_t<int8_t, int8_t, int32_t>;

} // namespace ppc64
} // namespace cpu
} // namespace impl
} // namespace dnnl
