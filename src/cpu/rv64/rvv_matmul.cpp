/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/gemm/rvv_gemm_f32.hpp"
#include "cpu/rv64/gemm/rvv_gemm_s8s8s32.hpp"
#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"
#include "cpu/rv64/rvv_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

void rvv_matmul_t::pd_t::init_scratchpad() {
    using namespace memory_tracking::names;
    using gemm_utils::calc_nthr_nocopy_rvv;
    using gemm_utils::gemm_utils_traits;

    // Size the scratchpad with max_threads at init, then hand this same
    // partition to the GEMM drivers at execute() (via gemm_partition_t) so the
    // per-thread workspace offsets never exceed the booked capacity — even when
    // init and execute run under different threadpool contexts. The drivers
    // only recompute from dnnl_get_current_num_threads() when no partition is
    // supplied (the inner_product / convolution callers).
    const int max_nthr = dnnl_get_max_threads();
    const dim_t gemm_M = N_;
    const dim_t gemm_N = weights_are_broadcast_ ? (batch_ * M_) : M_;
    calc_nthr_nocopy_rvv(gemm_M, gemm_N, K_, max_nthr, &nthr_m_, &nthr_n_,
            &nthr_k_, &MB_, &NB_, &KB_);

    // Pick the path's own m_unroll: f32 and int8 derive the factor independently
    m_unroll_ = is_f32_path_ ? gemm_utils_traits<float>::get_m_unroll_factor()
                             : gemm_utils_traits<int8_t>::get_m_unroll_factor();
    do_copy_ = (NB_ / gemm_utils_traits<float>::get_n_unroll_factor() > 3);

    const int nthr_mn = nthr_m_ * nthr_n_;
    const int nthr_to_use = nthr_mn * nthr_k_;

    auto scratchpad = scratchpad_registry().registrar();

    // K-split reduction buffer. Both f32 and s32 dst use 4-byte elements;
    // book<char> + byte count keeps the contract type-agnostic. Only needed
    // when the K axis is split across threads.
    if (nthr_k_ > 1) {
        const size_t c_elems
                = (size_t)nthr_m_ * nthr_n_ * (nthr_k_ - 1) * MB_ * NB_;
        scratchpad.template book<char>(key_gemm_accumulator, c_elems * 4);
    }

    // Per-thread A-copy workspace. Size mirrors what rvv_gemm_f32 /
    // rvv_gemm_s8s8s32 malloc in the fallback path: rnd_up(K*m_unroll*elem_size,
    // PAGE_4K) bytes per thread.
    if (do_copy_) {
        const size_t elem_size = is_f32_path_ ? sizeof(float) : sizeof(int8_t);
        const size_t ws_size_per_thr
                = utils::rnd_up((size_t)K_ * m_unroll_ * elem_size, PAGE_4K);
        scratchpad.template book<char>(
                key_gemm_tmp_buffer, (size_t)nthr_to_use * ws_size_per_thr);
    }
}

status_t rvv_matmul_t::init(engine_t *engine) {
    UNUSED(engine);
    // The int8 path has no post-op kernel yet; only build the per-row
    // "bias + post-op chain" kernel for the f32 dispatch.
    if (!pd()->is_int8_path_) {
        const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);
        jit_uni_postops_kernel_t::conf_t conf;
        conf.dst_dt = data_type::f32;
        conf.with_bias = !bias_d.is_zero();
        if (conf.with_bias) {
            const int bn = bias_d.ndims();
            // scalar when the whole bias is one value or its last dim
            // broadcasts over N; otherwise a per-N run aligned with the
            // output row.
            conf.bias_per_element
                    = !(bias_d.nelems() == 1 || bias_d.dims()[bn - 1] == 1);
        }
        return jit_uni_postops_kernel_t::create(
                postops_kernel_, pd()->attr()->post_ops_, conf);
    }
    return status::success;
}

namespace {
// GEMM M/N/K/lda/ldb/ldc setup shared by both the f32 and s8 paths. The
// driver reinterprets row-major [M, K] / [M, N] as column-major matrices and
// maps them to GEMM as C(N x M) = A(N x K) * B(K x M), with A = W^T and
// B = src (see the in-source comment in execute() for the full derivation).
struct gemm_axes_t {
    char transa;
    char transb;
    dim_t M_gemm;
    dim_t N_gemm;
    dim_t K_gemm;
    dim_t lda;
    dim_t ldb;
    dim_t ldc;
};

gemm_axes_t make_gemm_axes(dim_t M, dim_t N, dim_t K, bool weights_col_major) {
    gemm_axes_t g;
    g.transa = weights_col_major ? 'T' : 'N';
    g.transb = 'N';
    g.M_gemm = N;
    g.N_gemm = M;
    g.K_gemm = K;
    g.lda = weights_col_major ? K : N;
    g.ldb = K;
    g.ldc = N;
    return g;
}
} // namespace

status_t rvv_matmul_t::execute(const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);

    const int ndims = src_d.ndims();
    const int wei_ndims = weights_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();

    const dim_t batch = pd()->batch_;
    const dim_t M = pd()->M_;
    const dim_t K = pd()->K_;
    const dim_t N = pd()->N_;
    const bool weights_col_major = pd()->weights_col_major_;

    const auto g = make_gemm_axes(M, N, K, weights_col_major);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Reuse the thread partition booked in the scratchpad at init (sized with
    // dnnl_get_max_threads()) so the drivers' per-thread workspace offsets
    // never exceed the booked capacity, even when init and execute run under
    // different threadpool contexts.
    const gemm_utils::gemm_partition_t part {pd()->nthr_m_, pd()->nthr_n_,
            pd()->nthr_k_, pd()->MB_, pd()->NB_, pd()->KB_};

    const int src_batch_ndims = ndims > 2 ? ndims - 2 : 0;
    const int wei_batch_ndims = wei_ndims > 2 ? wei_ndims - 2 : 0;
    const int batch_dim_shift = src_batch_ndims - wei_batch_ndims;
    const dim_t K_dim = wei_dims[wei_ndims - 2];
    const dim_t N_dim = wei_dims[wei_ndims - 1];
    const dim_t wei_matrix_stride = K_dim * N_dim;

    if (pd()->is_int8_path_) {
        // Int8 dispatch: s8 weights * (s8|u8) src -> (s32|f32) dst. No
        // post-ops or scales (those attrs are rejected in pd_t::init), so the
        // only epilogue work is the optional fused bias inside the GEMM
        // kernel itself.
        const int8_t *weights = static_cast<const int8_t *>(
                CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS));
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        const float *bias = bias_d.is_zero()
                ? nullptr
                : CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
        // A scalar (one-element) bias broadcasts over the M tile: tell the
        // kernel to splat one float instead of reading a full vector.
        const bool bias_is_scalar = bias && bias_d.nelems() == 1;

        const bool b_signed = src_d.data_type() == data_type::s8;
        const bool dst_is_f32 = dst_d.data_type() == data_type::f32;

        const dim_t src_batch_stride = M * K;
        const dim_t dst_batch_stride = M * N;

        // Scratchpad buffers booked in pd_t::init_scratchpad().
        auto &grantor = ctx.get_scratchpad_grantor();
        int32_t *c_buffer = pd()->nthr_k_ > 1
                ? grantor.template get<int32_t>(
                          memory_tracking::names::key_gemm_accumulator)
                : nullptr;
        int8_t *ws_buffer = pd()->do_copy_
                ? grantor.template get<int8_t>(
                          memory_tracking::names::key_gemm_tmp_buffer)
                : nullptr;

        if (pd()->weights_are_broadcast_) {
            //   C(N x (batch * M)) = A(N x K) * B(K x (batch * M))
            const dim_t M_gemm_all = g.M_gemm; // N
            const dim_t N_gemm_all = batch * g.N_gemm; // batch * M

            status_t st = rvv_gemm_s8s8s32(&g.transa, &g.transb, &M_gemm_all,
                    &N_gemm_all, &g.K_gemm, &alpha, weights, &g.lda, src,
                    &g.ldb, &beta, dst, &g.ldc, /*bias=*/bias, b_signed,
                    dst_is_f32, c_buffer, ws_buffer, bias_is_scalar, &part);
            assert(st == status::success || st == status::unimplemented);
            MAYBE_UNUSED(st);
        } else {
            for (dim_t b = 0; b < batch; ++b) {
                const char *src_base = static_cast<const char *>(src)
                        + b * src_batch_stride
                                * types::data_type_size(src_d.data_type());
                char *dst_base = static_cast<char *>(dst)
                        + b * dst_batch_stride
                                * types::data_type_size(dst_d.data_type());

                dim_t batch_indices[DNNL_MAX_NDIMS] = {};
                if (src_batch_ndims > 0) {
                    utils::l_dims_by_l_offset(
                            batch_indices, b, src_dims, src_batch_ndims);
                }

                dim_t weight_batch_index = 0;
                if (wei_batch_ndims > 0) {
                    for (int d = 0; d < wei_batch_ndims; ++d) {
                        const int src_dim_idx = d + batch_dim_shift;
                        dim_t idx = (src_dim_idx >= 0)
                                ? batch_indices[src_dim_idx]
                                : dim_t(0);
                        const dim_t wei_dim = wei_dims[d];
                        idx = (wei_dim == 1) ? dim_t(0) : idx;
                        weight_batch_index = weight_batch_index * wei_dim + idx;
                    }
                }

                const int8_t *wei_base
                        = weights + weight_batch_index * wei_matrix_stride;

                status_t st = rvv_gemm_s8s8s32(&g.transa, &g.transb, &g.M_gemm,
                        &g.N_gemm, &g.K_gemm, &alpha, wei_base, &g.lda,
                        src_base, &g.ldb, &beta, dst_base, &g.ldc,
                        /*bias=*/bias, b_signed, dst_is_f32, c_buffer,
                        ws_buffer, bias_is_scalar, &part);
                assert(st == status::success || st == status::unimplemented);
                MAYBE_UNUSED(st);
            }
        }
        return status::success;
    }

    // f32 dispatch (unchanged): bias + post-op chain is applied per output row
    // after the GEMM by jit_uni_postops_kernel_t.
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    const float *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);

    // row-major [M, K] / [M, N] are reinterpreted as column-major matrices and
    // mapped to GEMM as:
    //   C(N x M) = A(N x K) * B(K x M)
    // where:
    //   - A = W^T, with W logical layout [K, N] row-major. Since row-major
    //     [K, N] is equivalent in memory to column-major [N, K] (leading dim N),
    //     we pass transa = 'N' and lda = N.
    //   - B = src, where src logical layout [M, K] row-major is equivalent to
    //     column-major [K, M] with leading dim K, so transb = 'N', ldb = K.
    //   - C is viewed as column-major [N, M] with leading dim N, which matches
    //     row-major [M, N] in memory.
    const dim_t src_batch_stride = M * K;
    const dim_t dst_batch_stride = M * N;

    auto &grantor = ctx.get_scratchpad_grantor();
    float *c_buffer = pd()->nthr_k_ > 1
            ? grantor.template get<float>(
                      memory_tracking::names::key_gemm_accumulator)
            : nullptr;
    float *ws_buffer = pd()->do_copy_
            ? grantor.template get<float>(
                      memory_tracking::names::key_gemm_tmp_buffer)
            : nullptr;

    if (pd()->weights_are_broadcast_) {
        //   C(N x (batch * M)) = A(N x K) * B(K x (batch * M))
        dim_t M_gemm_all = g.M_gemm; // N
        dim_t N_gemm_all = batch * g.N_gemm; // batch * M

        status_t st = rvv_gemm_f32(&g.transa, &g.transb, &M_gemm_all,
                &N_gemm_all, &g.K_gemm, &alpha, weights, &g.lda, src, &g.ldb,
                &beta, dst, &g.ldc,
                /*bias=*/nullptr, c_buffer, ws_buffer, &part);
        assert(st == status::success || st == status::unimplemented);
        MAYBE_UNUSED(st);

    } else {
        for (dim_t b = 0; b < batch; ++b) {
            const float *src_base = src + b * src_batch_stride;
            float *dst_base = dst + b * dst_batch_stride;

            dim_t batch_indices[DNNL_MAX_NDIMS] = {};
            if (src_batch_ndims > 0) {
                utils::l_dims_by_l_offset(
                        batch_indices, b, src_dims, src_batch_ndims);
            }

            dim_t weight_batch_index = 0;
            if (wei_batch_ndims > 0) {
                for (int d = 0; d < wei_batch_ndims; ++d) {
                    const int src_dim_idx = d + batch_dim_shift;
                    dim_t idx = (src_dim_idx >= 0) ? batch_indices[src_dim_idx]
                                                   : dim_t(0);
                    const dim_t wei_dim = wei_dims[d];
                    idx = (wei_dim == 1) ? dim_t(0) : idx;
                    weight_batch_index = weight_batch_index * wei_dim + idx;
                }
            }

            const float *wei_base
                    = weights + weight_batch_index * wei_matrix_stride;

            status_t st = rvv_gemm_f32(&g.transa, &g.transb, &g.M_gemm,
                    &g.N_gemm, &g.K_gemm, &alpha, wei_base, &g.lda, src_base,
                    &g.ldb, &beta, dst_base, &g.ldc,
                    /*bias=*/nullptr, c_buffer, ws_buffer, &part);
            assert(st == status::success || st == status::unimplemented);
            MAYBE_UNUSED(st);
        }
    }

    if (!bias && post_ops.len() == 0) return status::success;

    const int dst_ndims = dst_d.ndims();
    const int bias_ndims = bias_d.ndims();
    const dim_t *bias_dims = bias_d.dims();

    // Binary post-op src1 bases, one per binary in chain order (per-N or scalar;
    // each broadcasts over M/batch so the same array serves every row). Empty
    // when the chain has no binary entry. Shift each base by src1's own offset0
    // (off_l(0)) so a submemory rhs is read from its logical origin, not the
    // buffer base — the kernel only adds the in-row column offset on top.
    std::vector<const void *> po_rhs;
    for (int i = 0; i < post_ops.len(); i++)
        if (post_ops.entry_[i].is_binary()) {
            const memory_desc_wrapper s1_d(post_ops.entry_[i].binary.src1_desc);
            const auto *base = static_cast<const char *>(ctx.host_ptr(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1));
            po_rhs.push_back(base + s1_d.off_l(0) * sizeof(float));
        }
    const void *const *po_rhs_arr = po_rhs.empty() ? nullptr : po_rhs.data();

    parallel_nd(batch, [&](dim_t b) {
        float *dst_base = dst + b * dst_batch_stride;

        dim_t dst_idx_prefix[DNNL_MAX_NDIMS] = {};
        size_t bias_strides[DNNL_MAX_NDIMS] = {};

        if (bias && bias_ndims > 1) {
            bias_strides[bias_ndims - 1] = 1;
            for (int d = bias_ndims - 2; d >= 0; --d)
                bias_strides[d]
                        = bias_strides[d + 1] * (size_t)bias_dims[d + 1];
        }

        for (dim_t m = 0; m < M; ++m) {
            if (ndims > 2) {
                utils::l_dims_by_l_offset(
                        dst_idx_prefix, b, src_dims, ndims - 2);
            }
            dst_idx_prefix[ndims - 2] = m;

            float *row_dst = dst_base + m * N;

            const float *bias_ptr = nullptr;
            if (bias) {
                if (bias_d.nelems() == 1) {
                    bias_ptr = bias;
                } else {
                    size_t base_bias_off = 0;
                    if (bias_ndims > 1) {
                        for (int d = 0; d < bias_ndims - 1; ++d) {
                            int dst_dim_idx = d + (dst_ndims - bias_ndims);
                            dim_t idx = (bias_dims[d] == 1)
                                    ? 0
                                    : dst_idx_prefix[dst_dim_idx];
                            base_bias_off += idx * bias_strides[d];
                        }
                    }
                    bias_ptr = bias + base_bias_off;
                }
            }

            // Fused bias + post-op chain over this output row (length N).
            jit_uni_postops_kernel_t::call_params_t cp;
            cp.dst = row_dst;
            cp.bias = bias_ptr;
            cp.rhs = po_rhs_arr;
            cp.off0 = 0; // per-N rhs starts at column 0 of every row
            cp.len = N;
            (*postops_kernel_)(&cp);
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
