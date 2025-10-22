/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_eltwise.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

#define DATA_OFF(f, n, c, d, h, w) \
    (ndims == 1) \
            ? (f).off(n) \
            : ((ndims == 2) ? (f).off(n, c) \
                            : ((ndims == 3) ? (f).off(n, c, w) \
                                            : ((ndims == 4) ? (f).off( \
                                                       n, c, h, w) \
                                                            : (f).off(n, c, d, \
                                                                    h, w))))

status_t ref_eltwise_fwd_t::execute_forward_generic(
        const exec_ctx_t &ctx) const {
    /* fast return */
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const int ndims = pd()->ndims();

    parallel_nd(
            MB, C, D, H, W, [=](dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
                auto data_p_off = DATA_OFF(src_d, n, c, d, h, w);
                const float s = io::load_float_value(
                        src_d.data_type(), src, data_p_off);
                float res
                        = compute_eltwise_scalar_fwd(alg_kind, s, alpha, beta);
                dim_t data_l_off = (((n * C + c) * D + d) * H + h) * W + w;

                ref_post_ops_t::args_t args;
                args.ctx = &ctx;
                args.l_offset = data_l_off;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(res, args);

                io::store_float_value(dst_d.data_type(), res, dst, data_p_off);
            });
    return status::success;
}

status_t ref_eltwise_fwd_t::execute_forward_dense(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto nelems = src_d.nelems(true);
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    src = static_cast<const char *>(src)
            + src_d.data_type_size() * src_d.offset0();
    dst = static_cast<char *>(dst) + dst_d.data_type_size() * dst_d.offset0();

    // a fast path for relu as the most popular activation
    if (alg_kind == alg_kind::eltwise_relu && alpha == 0) {
        parallel_nd(nelems, [=](dim_t e) {
            const float s = io::load_float_value(src_d.data_type(), src, e);
            float res = math::relu_fwd(s, alpha);
            io::store_float_value(dst_d.data_type(), res, dst, e);
        });
        return status::success;
    }

    parallel_nd(nelems, [=](dim_t e) {
        const float s = io::load_float_value(src_d.data_type(), src, e);
        float res = compute_eltwise_scalar_fwd(alg_kind, s, alpha, beta);
        io::store_float_value(dst_d.data_type(), res, dst, e);
    });
    return status::success;
}

status_t ref_eltwise_bwd_t::execute_backward_generic(
        const exec_ctx_t &ctx) const {
    /* fast return */
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;
    auto src = CTX_IN_MEM(
            const void *, pd()->use_dst() ? DNNL_ARG_DST : DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const int ndims = pd()->ndims();

    parallel_nd(
            MB, C, D, H, W, [=](dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
                auto data_off = DATA_OFF(data_d, n, c, d, h, w);
                auto diff_data_off = DATA_OFF(diff_data_d, n, c, d, h, w);
                const float s = io::load_float_value(
                        data_d.data_type(), src, data_off);
                const float dd = io::load_float_value(
                        diff_data_d.data_type(), diff_dst, diff_data_off);
                float res = compute_eltwise_scalar_bwd(
                        alg_kind, dd, s, alpha, beta);
                io::store_float_value(
                        diff_data_d.data_type(), res, diff_src, diff_data_off);
            });
    return status::success;
}

status_t ref_eltwise_bwd_t::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    const void *src = pd()->use_dst() ? CTX_IN_MEM(const void *, DNNL_ARG_DST)
                                      : CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    void *diff_src = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());

    const auto nelems = data_d.nelems(true);
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    src = static_cast<const char *>(src)
            + data_d.data_type_size() * data_d.offset0();
    diff_dst = static_cast<const char *>(diff_dst)
            + diff_data_d.data_type_size() * diff_data_d.offset0();
    diff_src = static_cast<char *>(diff_src)
            + diff_data_d.data_type_size() * diff_data_d.offset0();

    if (data_d.data_type() == data_type::f32) {
        parallel(0, [=](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            for (dim_t i = start; i < end; i++) {
                const float s
                        = io::load_float_value(data_d.data_type(), src, i);
                const float dd = io::load_float_value(
                        diff_data_d.data_type(), diff_dst, i);
                float res = compute_eltwise_scalar_bwd(
                        alg_kind, dd, s, alpha, beta);
                io::store_float_value(
                        diff_data_d.data_type(), res, diff_src, i);
            }
        });
    } else if (utils::one_of(data_d.data_type(), data_type::bf16,
                       data_type::f16, data_type::f8_e5m2,
                       data_type::f8_e4m3)) {
        using namespace memory_tracking::names;
        const auto &scratchpad = ctx.get_scratchpad_grantor();
        auto *src_f32 = scratchpad.template get<float>(key_eltwise_src);
        auto *diff_dst_f32
                = scratchpad.template get<float>(key_eltwise_diff_dst);

        parallel(0, [=](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            types::cvt_to_float(data_d.data_type(), src_f32 + start,
                    static_cast<const char *>(src)
                            + data_d.data_type_size() * start,
                    end - start);
            types::cvt_to_float(diff_data_d.data_type(), diff_dst_f32 + start,
                    static_cast<const char *>(diff_dst)
                            + diff_data_d.data_type_size() * start,
                    end - start);

            for (dim_t i = start; i < end; i++) {
                diff_dst_f32[i] = compute_eltwise_scalar_bwd(
                        alg_kind, diff_dst_f32[i], src_f32[i], alpha, beta);
            }

            types::cvt_from_float(data_d.data_type(),
                    static_cast<char *>(diff_src)
                            + diff_data_d.data_type_size() * start,
                    diff_dst_f32 + start, end - start);
        });
    } else {
        assert(!"unsupported data type");
    }
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
