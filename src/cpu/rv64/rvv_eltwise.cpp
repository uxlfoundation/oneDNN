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

#include "cpu/rv64/rvv_eltwise.hpp"
#include "cpu/rv64/rvv_eltwise_kernels.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

// Data type dispatch for RVV eltwise forward
void compute_eltwise_rvv_fwd(const alg_kind_t alg, const void *src, void *dst,
        const float alpha, const float beta, const dim_t len,
        const data_type_t dt) {
    switch (dt) {
        case data_type::f32:
            rvv_eltwise_apply_fwd_f32(alg, src, dst, len, alpha, beta, dt);
            break;
        case data_type::s32:
            rvv_eltwise_apply_fwd_s32(alg, src, dst, len, alpha, beta, dt);
            break;
        case data_type::s8:
            rvv_eltwise_apply_fwd_s8(alg, src, dst, len, alpha, beta, dt);
            break;
        case data_type::u8:
            rvv_eltwise_apply_fwd_u8(alg, src, dst, len, alpha, beta, dt);
            break;
        default: assert(!"Unsupported data type for RVV eltwise");
    }
}

// Data type dispatch for RVV eltwise backward
void compute_eltwise_rvv_bwd(const alg_kind_t alg, void *diff_src,
        const void *diff_dst, const void *src, const float alpha,
        const float beta, const dim_t len, const data_type_t dt) {
    switch (dt) {
        case data_type::f32:
            rvv_eltwise_apply_bwd_f32(
                    alg, diff_src, diff_dst, src, len, alpha, beta, dt);
            break;
        case data_type::s32:
            rvv_eltwise_apply_bwd_s32(
                    alg, diff_src, diff_dst, src, len, alpha, beta, dt);
            break;
        case data_type::s8:
            rvv_eltwise_apply_bwd_s8(
                    alg, diff_src, diff_dst, src, len, alpha, beta, dt);
            break;
        case data_type::u8:
            rvv_eltwise_apply_bwd_u8(
                    alg, diff_src, diff_dst, src, len, alpha, beta, dt);
            break;
        default: assert(!"Unsupported data type for RVV eltwise");
    }
}

} // unnamed namespace

// Forward execute
status_t rvv_eltwise_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;
    const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    void *dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto nelems = dst_d.nelems(true);

    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    if (pd()->use_dense_) {
        const size_t esize = types::data_type_size(pd()->src_md()->data_type);
        const char *src_base
                = static_cast<const char *>(src) + src_d.offset0() * esize;
        char *dst_base = static_cast<char *>(dst) + dst_d.offset0() * esize;

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            const void *thr_src
                    = static_cast<const void *>(src_base + start * esize);
            void *thr_dst = static_cast<void *>(dst_base + start * esize);
            const dim_t len = end - start;

            compute_eltwise_rvv_fwd(alg_kind, thr_src, thr_dst, alpha, beta,
                    len, pd()->src_md()->data_type);
        });
    }
    return status::success;
}

// Backward execute
status_t rvv_eltwise_bwd_t::execute(const exec_ctx_t &ctx) const {
    auto data = pd()->use_dst() ? CTX_IN_MEM(const void *, DNNL_ARG_DST)
                                : CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto nelems = diff_src_d.nelems(true);
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    if (pd()->use_dense_) {
        const size_t esize = types::data_type_size(pd()->src_md()->data_type);
        const dim_t off = diff_src_d.offset0();
        char *ds_bytes = static_cast<char *>(diff_src) + off * esize;
        const char *dd_bytes
                = static_cast<const char *>(diff_dst) + off * esize;
        const char *data_bytes = static_cast<const char *>(data) + off * esize;

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            compute_eltwise_rvv_bwd(alg_kind,
                    static_cast<void *>(ds_bytes + start * esize),
                    static_cast<const void *>(dd_bytes + start * esize),
                    static_cast<const void *>(data_bytes + start * esize),
                    alpha, beta, end - start, pd()->src_md()->data_type);
        });
    }
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
