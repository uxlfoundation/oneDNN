/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include <assert.h>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/primitive_attr_postops.hpp"
#include "cpu/rv64/rvv_eltwise_kernels.hpp"

#include "cpu/rv64/rvv_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Data type dispatch for RVV eltwise forward (per-dtype apply)
static inline void compute_eltwise_rvv_fwd(const alg_kind_t alg,
        const void *src, void *dst, const float alpha, const float beta,
        const dim_t len, const data_type_t dt) {
    switch (dt) {
        case data_type::f32:
            rvv_eltwise_apply_fwd_f32(alg, reinterpret_cast<const float *>(src),
                    reinterpret_cast<float *>(dst), len, alpha, beta);
            break;
        case data_type::f16:
            rvv_eltwise_apply_fwd_f16(alg,
                    reinterpret_cast<const _Float16 *>(src),
                    reinterpret_cast<_Float16 *>(dst), len, alpha, beta);
            break;
        case data_type::s32:
            rvv_eltwise_apply_fwd_s32(alg,
                    reinterpret_cast<const int32_t *>(src),
                    reinterpret_cast<int32_t *>(dst), len, alpha, beta);
            break;
        case data_type::s8:
            rvv_eltwise_apply_fwd_s8(alg, reinterpret_cast<const int8_t *>(src),
                    reinterpret_cast<int8_t *>(dst), len, alpha, beta);
            break;
        case data_type::u8:
            rvv_eltwise_apply_fwd_u8(alg,
                    reinterpret_cast<const uint8_t *>(src),
                    reinterpret_cast<uint8_t *>(dst), len, alpha, beta);
            break;
        default: assert(!"Unsupported data type for RVV eltwise");
    }
}

// Data type dispatch for RVV eltwise backward (per-dtype apply)
static inline void compute_eltwise_rvv_bwd(const alg_kind_t alg, void *diff_src,
        const void *diff_dst, const void *src, const float alpha,
        const float beta, const dim_t len, const data_type_t dt) {
    switch (dt) {
        case data_type::f32:
            rvv_eltwise_apply_bwd_f32(alg, reinterpret_cast<float *>(diff_src),
                    reinterpret_cast<const float *>(diff_dst),
                    reinterpret_cast<const float *>(src), len, alpha, beta);
            break;
        case data_type::f16:
            rvv_eltwise_apply_bwd_f16(alg,
                    reinterpret_cast<_Float16 *>(diff_src),
                    reinterpret_cast<const _Float16 *>(diff_dst),
                    reinterpret_cast<const _Float16 *>(src), len, alpha, beta);
            break;
        case data_type::s32:
            rvv_eltwise_apply_bwd_s32(alg,
                    reinterpret_cast<int32_t *>(diff_src),
                    reinterpret_cast<const int32_t *>(diff_dst),
                    reinterpret_cast<const int32_t *>(src), len, alpha, beta);
            break;
        case data_type::s8:
            rvv_eltwise_apply_bwd_s8(alg, reinterpret_cast<int8_t *>(diff_src),
                    reinterpret_cast<const int8_t *>(diff_dst),
                    reinterpret_cast<const int8_t *>(src), len, alpha, beta);
            break;
        case data_type::u8:
            rvv_eltwise_apply_bwd_u8(alg, reinterpret_cast<uint8_t *>(diff_src),
                    reinterpret_cast<const uint8_t *>(diff_dst),
                    reinterpret_cast<const uint8_t *>(src), len, alpha, beta);
            break;
        default: assert(!"Unsupported data type for RVV eltwise");
    }
}

// Forward execute
template <data_type_t data_type>
status_t rvv_eltwise_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto nelems = data_d.nelems(true);

    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    if (pd()->use_dense_) {
        src += data_d.offset0();
        dst += data_d.offset0();

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            const void *thr_src = static_cast<const void *>(src + start);
            void *thr_dst = static_cast<void *>(dst + start);
            const dim_t len = end - start;

            compute_eltwise_rvv_fwd(alg_kind, thr_src, thr_dst, alpha, beta,
                    len, pd()->src_md()->data_type);
        });

        return status::success;
    }

    // nCspBc padded path: iterate over blocks and handle tail with zero-preserve
    if (pd()->use_nCspBc_padded_) {
        const blocking_desc_t &blk = data_d.blocking_desc();
        const dim_t block = blk.inner_blks[0];

        const dim_t MB = pd()->MB();
        const dim_t C = pd()->C() / block;
        const dim_t C_PADDED = data_d.padded_dims()[1] / block;
        const dim_t tail = pd()->C() % block;
        const dim_t SP = pd()->D() * pd()->H() * pd()->W();

        parallel_nd(MB, C_PADDED, SP, [&](dim_t n, dim_t c, dim_t sp) {
            auto d_off = (n * C_PADDED * SP + c * SP + sp) * block;

            const void *thr_src = static_cast<const void *>(src + d_off);
            void *thr_dst = static_cast<void *>(dst + d_off);

            if (c < C) {
                // full block
                compute_eltwise_rvv_fwd(alg_kind, thr_src, thr_dst, alpha, beta,
                        block, pd()->src_md()->data_type);
            } else {
                // tail: process only valid channels, keep padding zero-preserved
                const dim_t len = tail;
                if (len > 0) {
                    compute_eltwise_rvv_fwd(alg_kind, thr_src, thr_dst, alpha,
                            beta, len, pd()->src_md()->data_type);
                }
            }
        });

        return status::success;
    }

    return status::unimplemented;
}

// Backward execute
template <data_type_t data_type>
status_t rvv_eltwise_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;
    auto data = pd()->use_dst() ? CTX_IN_MEM(const data_t *, DNNL_ARG_DST)
                                : CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_d(pd()->diff_src_md());

    const auto nelems = diff_d.nelems(true);
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    if (pd()->use_dense_) {
        const dim_t off = diff_d.offset0();
        data_t *ds_ptr = diff_src + off;
        const data_t *dd_ptr = diff_dst + off;
        const data_t *data_ptr = data + off;

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            compute_eltwise_rvv_bwd(alg_kind,
                    static_cast<void *>(ds_ptr + start),
                    static_cast<const void *>(dd_ptr + start),
                    static_cast<const void *>(data_ptr + start), alpha, beta,
                    end - start, pd()->src_md()->data_type);
        });
        return status::success;
    }

    if (pd()->use_nCspBc_padded_) {
        const blocking_desc_t &blk = data_d.blocking_desc();
        const dim_t block = blk.inner_blks[0];

        const dim_t MB = pd()->MB();
        const dim_t C = pd()->C() / block;
        const dim_t C_PADDED = data_d.padded_dims()[1] / block;
        const dim_t tail = pd()->C() % block;
        const dim_t SP = pd()->D() * pd()->H() * pd()->W();

        parallel_nd(MB, C_PADDED, SP, [&](dim_t n, dim_t c, dim_t sp) {
            auto base_off = (n * C_PADDED * SP + c * SP + sp) * block;

            auto data_p = data + base_off;
            auto dd_p = diff_dst + base_off;
            auto ds_p = diff_src + base_off;

            const dim_t len = (c < C) ? block : tail;
            if (len == 0) return;
            compute_eltwise_rvv_bwd(alg_kind, static_cast<void *>(ds_p),
                    static_cast<const void *>(dd_p),
                    static_cast<const void *>(data_p), alpha, beta, len,
                    pd()->src_md()->data_type);
        });
        return status::success;
    }

    return status::unimplemented;
}

template struct rvv_eltwise_fwd_t<data_type::f32>;
template struct rvv_eltwise_fwd_t<data_type::f16>;
template struct rvv_eltwise_fwd_t<data_type::s32>;
template struct rvv_eltwise_fwd_t<data_type::s8>;
template struct rvv_eltwise_fwd_t<data_type::u8>;

template struct rvv_eltwise_bwd_t<data_type::f32>;
template struct rvv_eltwise_bwd_t<data_type::f16>;
template struct rvv_eltwise_bwd_t<data_type::s32>;
template struct rvv_eltwise_bwd_t<data_type::s8>;
template struct rvv_eltwise_bwd_t<data_type::u8>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl