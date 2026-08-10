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
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "cpu/binary_injector_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/gemm/rvv_gemm_f16.hpp"
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

    // Pick the path's own m_unroll: f32 and int8 derive the factor
    // independently; the half-precision kernel loads A at e16/m2, which holds
    // the same element count as the f32 kernel's e32/m4.
    m_unroll_ = is_int8_path_ ? gemm_utils_traits<int8_t>::get_m_unroll_factor()
                              : gemm_utils_traits<float>::get_m_unroll_factor();
    do_copy_ = (NB_ / gemm_utils_traits<float>::get_n_unroll_factor() > 3);

    const int nthr_mn = nthr_m_ * nthr_n_;
    const int nthr_to_use = nthr_mn * nthr_k_;

    auto scratchpad = scratchpad_registry().registrar();

    // K-split reduction buffer. Both f32 and s32 dst use 4-byte elements;
    // book<char> + byte count keeps the contract type-agnostic. Only needed
    // when the K axis is split across threads; the half-precision GEMM never
    // splits K (overwrite-only epilogue).
    if (nthr_k_ > 1 && !is_hp_path_) {
        const size_t c_elems
                = (size_t)nthr_m_ * nthr_n_ * (nthr_k_ - 1) * MB_ * NB_;
        scratchpad.template book<char>(key_gemm_accumulator, c_elems * 4);
    }

    // Per-thread A-copy workspace. Size mirrors what the GEMM drivers malloc
    // in the fallback path: rnd_up(K*m_unroll*elem_size, PAGE_4K) bytes per
    // thread. With per-batch weights, execute() may split the batch loop
    // across workers, each driving a single-thread GEMM partition whose NB is
    // the full gemm_N — the copy may trigger there even when the full
    // partition's NB_ says otherwise, so cover both thread counts.
    const dim_t batch_workers
            = weights_are_broadcast_ ? 0 : nstl::min(batch_, (dim_t)max_nthr);
    ws_thr_slices_
            = static_cast<int>(nstl::max<dim_t>(nthr_to_use, batch_workers));
    const bool per_batch_copy
            = M_ / gemm_utils_traits<float>::get_n_unroll_factor() > 3;
    if (do_copy_ || (batch_workers > 0 && per_batch_copy)) {
        const size_t elem_size = is_int8_path_ ? sizeof(int8_t)
                : is_hp_path_                  ? 2 // f16/bf16 elements
                                               : sizeof(float);
        ws_slice_bytes_
                = utils::rnd_up((size_t)K_ * m_unroll_ * elem_size, PAGE_4K);
        scratchpad.template book<char>(
                key_gemm_tmp_buffer, (size_t)ws_thr_slices_ * ws_slice_bytes_);
    }
}

status_t rvv_matmul_t::init(engine_t *engine) {
    UNUSED(engine);
    // Only the f32 dispatch applies the per-row "bias + post-op chain"
    // kernel (it hardcodes an f32 dst); the int8 path fuses bias inside its
    // GEMM kernel and the half-precision path has no bias / post-ops yet.
    if (pd()->is_f32_path_) {
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

// Runs `call_one(b, ws, part)` for every batch element. With enough batch
// entries the loop runs in parallel, each worker driving a single-thread GEMM
// partition over its entries with its own A-copy workspace slice; otherwise
// the batch is walked sequentially with the full partition. This makes the
// batch dimension part of the parallel work for per-batch weights, like the
// aarch64/x64 brgemm matmuls do, instead of an outer serial loop.
template <typename F>
void run_batch_gemm(dim_t batch, int ws_thr_slices, size_t ws_slice_bytes,
        char *ws_base, const gemm_utils::gemm_partition_t *part_full,
        const gemm_utils::gemm_partition_t *part_single, F call_one) {
    dim_t nworkers = nstl::min(batch, (dim_t)dnnl_get_current_num_threads());
    nworkers = nstl::min(nworkers, (dim_t)ws_thr_slices);
    if (nworkers > 1) {
        parallel(static_cast<int>(nworkers), [&](int ithr, int nthr) {
            char *ws = ws_base ? ws_base + ithr * ws_slice_bytes : nullptr;
            for (dim_t b = ithr; b < batch; b += nthr)
                call_one(b, ws, part_single);
        });
    } else {
        for (dim_t b = 0; b < batch; ++b)
            call_one(b, ws_base, part_full);
    }
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
    // Single-thread partition for the parallel batch loop (run_batch_gemm).
    const gemm_utils::gemm_partition_t part_single {
            1, 1, 1, g.M_gemm, g.N_gemm, g.K_gemm};

    const int src_batch_ndims = ndims > 2 ? ndims - 2 : 0;
    const int wei_batch_ndims = wei_ndims > 2 ? wei_ndims - 2 : 0;
    const int batch_dim_shift = src_batch_ndims - wei_batch_ndims;
    const dim_t K_dim = wei_dims[wei_ndims - 2];
    const dim_t N_dim = wei_dims[wei_ndims - 1];
    const dim_t wei_matrix_stride = K_dim * N_dim;

    // Byte views of the operands so the per-dtype dispatches share one batch
    // addressing scheme (strides in bytes, per-dtype element sizes).
    const char *src_bytes
            = static_cast<const char *>(CTX_IN_MEM(const void *, DNNL_ARG_SRC));
    const char *wei_bytes = static_cast<const char *>(
            CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS));
    char *dst_bytes = static_cast<char *>(CTX_OUT_MEM(void *, DNNL_ARG_DST));
    const dim_t src_batch_stride_bytes
            = M * K * types::data_type_size(src_d.data_type());
    const dim_t dst_batch_stride_bytes
            = M * N * types::data_type_size(dst_d.data_type());

    // Weights base for batch element b, honoring broadcast (size-1) batch
    // dims. Local scratch: called from parallel workers.
    auto wei_base = [&](dim_t b) -> const char * {
        if (wei_batch_ndims == 0) return wei_bytes;
        dim_t batch_indices[DNNL_MAX_NDIMS] = {};
        utils::l_dims_by_l_offset(batch_indices, b, src_dims, src_batch_ndims);
        dim_t weight_batch_index = 0;
        for (int d = 0; d < wei_batch_ndims; ++d) {
            const int src_dim_idx = d + batch_dim_shift;
            dim_t idx = (src_dim_idx >= 0) ? batch_indices[src_dim_idx]
                                           : dim_t(0);
            const dim_t wei_dim = wei_dims[d];
            idx = (wei_dim == 1) ? dim_t(0) : idx;
            weight_batch_index = weight_batch_index * wei_dim + idx;
        }
        return wei_bytes
                + weight_batch_index * wei_matrix_stride
                * types::data_type_size(weights_d.data_type());
    };

    auto &grantor = ctx.get_scratchpad_grantor();
    char *ws_bytes = pd()->ws_slice_bytes_
            ? grantor.template get<char>(
                      memory_tracking::names::key_gemm_tmp_buffer)
            : nullptr;

    if (pd()->is_int8_path_) {
        // Int8 dispatch: (s8|u8) weights * (s8|u8) src ->
        // (s32|f32|s8|u8|f16|bf16) dst. No post-ops or scales (those attrs are
        // rejected in pd_t::init), so the only epilogue work is the optional
        // fused bias inside the GEMM kernel itself.
        const float *bias = bias_d.is_zero()
                ? nullptr
                : CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
        // A scalar (one-element) bias broadcasts over the M tile: tell the
        // kernel to splat one float instead of reading a full vector.
        const bool bias_is_scalar = bias && bias_d.nelems() == 1;

        // A axis = weights, B axis = src (see the GEMM mapping comment
        // below). Each operand is independently s8/u8.
        const bool a_signed = weights_d.data_type() == data_type::s8;
        const bool b_signed = src_d.data_type() == data_type::s8;
        const data_type_t dst_dt = dst_d.data_type();

        int32_t *c_buffer = pd()->nthr_k_ > 1
                ? grantor.template get<int32_t>(
                          memory_tracking::names::key_gemm_accumulator)
                : nullptr;

        if (pd()->weights_are_broadcast_) {
            //   C(N x (batch * M)) = A(N x K) * B(K x (batch * M))
            const dim_t M_gemm_all = g.M_gemm; // N
            const dim_t N_gemm_all = batch * g.N_gemm; // batch * M

            status_t st = rvv_gemm_s8s8s32(&g.transa, &g.transb, &M_gemm_all,
                    &N_gemm_all, &g.K_gemm, &alpha, wei_bytes, &g.lda,
                    src_bytes, &g.ldb, &beta, dst_bytes, &g.ldc,
                    /*bias=*/bias, a_signed, b_signed, dst_dt, c_buffer,
                    reinterpret_cast<int8_t *>(ws_bytes), bias_is_scalar,
                    &part);
            assert(st == status::success || st == status::unimplemented);
            MAYBE_UNUSED(st);
        } else {
            run_batch_gemm(batch, pd()->ws_thr_slices_, pd()->ws_slice_bytes_,
                    ws_bytes, &part, &part_single,
                    [&](dim_t b, char *ws,
                            const gemm_utils::gemm_partition_t *p) {
                status_t st = rvv_gemm_s8s8s32(&g.transa, &g.transb, &g.M_gemm,
                        &g.N_gemm, &g.K_gemm, &alpha, wei_base(b), &g.lda,
                        src_bytes + b * src_batch_stride_bytes, &g.ldb, &beta,
                        dst_bytes + b * dst_batch_stride_bytes, &g.ldc,
                        /*bias=*/bias, a_signed, b_signed, dst_dt, c_buffer,
                        reinterpret_cast<int8_t *>(ws), bias_is_scalar, p);
                assert(st == status::success || st == status::unimplemented);
                MAYBE_UNUSED(st);
            });
        }
        return status::success;
    }

    if (pd()->is_hp_path_) {
        // Half-precision dispatch: (f16|bf16) weights * (f16|bf16) src ->
        // same-dtype dst with f32 accumulation inside the GEMM kernel. No
        // bias / post-ops on this path (rejected in pd_t::init).
        const data_type_t hp_dt = src_d.data_type();

        if (pd()->weights_are_broadcast_) {
            //   C(N x (batch * M)) = A(N x K) * B(K x (batch * M))
            const dim_t M_gemm_all = g.M_gemm; // N
            const dim_t N_gemm_all = batch * g.N_gemm; // batch * M

            status_t st = rvv_gemm_f16(&g.transa, &g.transb, &M_gemm_all,
                    &N_gemm_all, &g.K_gemm, &alpha, wei_bytes, &g.lda,
                    src_bytes, &g.ldb, &beta, dst_bytes, &g.ldc, hp_dt,
                    ws_bytes, &part);
            assert(st == status::success || st == status::unimplemented);
            MAYBE_UNUSED(st);
        } else {
            run_batch_gemm(batch, pd()->ws_thr_slices_, pd()->ws_slice_bytes_,
                    ws_bytes, &part, &part_single,
                    [&](dim_t b, char *ws,
                            const gemm_utils::gemm_partition_t *p) {
                status_t st = rvv_gemm_f16(&g.transa, &g.transb, &g.M_gemm,
                        &g.N_gemm, &g.K_gemm, &alpha, wei_base(b), &g.lda,
                        src_bytes + b * src_batch_stride_bytes, &g.ldb, &beta,
                        dst_bytes + b * dst_batch_stride_bytes, &g.ldc, hp_dt,
                        ws, p);
                assert(st == status::success || st == status::unimplemented);
                MAYBE_UNUSED(st);
            });
        }
        return status::success;
    }

    // f32 dispatch (unchanged): bias + post-op chain is applied per output row
    // after the GEMM by jit_uni_postops_kernel_t.
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
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
    const dim_t dst_batch_stride = M * N;

    float *c_buffer = pd()->nthr_k_ > 1
            ? grantor.template get<float>(
                      memory_tracking::names::key_gemm_accumulator)
            : nullptr;

    if (pd()->weights_are_broadcast_) {
        //   C(N x (batch * M)) = A(N x K) * B(K x (batch * M))
        dim_t M_gemm_all = g.M_gemm; // N
        dim_t N_gemm_all = batch * g.N_gemm; // batch * M

        status_t st = rvv_gemm_f32(&g.transa, &g.transb, &M_gemm_all,
                &N_gemm_all, &g.K_gemm, &alpha,
                reinterpret_cast<const float *>(wei_bytes), &g.lda, src, &g.ldb,
                &beta, dst, &g.ldc,
                /*bias=*/nullptr, c_buffer, reinterpret_cast<float *>(ws_bytes),
                &part);
        assert(st == status::success || st == status::unimplemented);
        MAYBE_UNUSED(st);

    } else {
        run_batch_gemm(batch, pd()->ws_thr_slices_, pd()->ws_slice_bytes_,
                ws_bytes, &part, &part_single,
                [&](dim_t b, char *ws, const gemm_utils::gemm_partition_t *p) {
            status_t st = rvv_gemm_f32(&g.transa, &g.transb, &g.M_gemm,
                    &g.N_gemm, &g.K_gemm, &alpha,
                    reinterpret_cast<const float *>(wei_base(b)), &g.lda,
                    reinterpret_cast<const float *>(
                            src_bytes + b * src_batch_stride_bytes),
                    &g.ldb, &beta,
                    reinterpret_cast<float *>(
                            dst_bytes + b * dst_batch_stride_bytes),
                    &g.ldc,
                    /*bias=*/nullptr, c_buffer, reinterpret_cast<float *>(ws),
                    p);
            assert(st == status::success || st == status::unimplemented);
            MAYBE_UNUSED(st);
        });
    }

    if (!bias && post_ops.len() == 0) return status::success;

    const int dst_ndims = dst_d.ndims();
    const int bias_ndims = bias_d.ndims();
    const dim_t *bias_dims = bias_d.dims();

    // Binary post-op src1 bases, one per binary in chain order (per-N or scalar;
    // each broadcasts over M/batch so the same array serves every row). Empty
    // when the chain has no binary entry. The raw handles from the common
    // helper (x64 model) -- the kernel only adds the in-row column offset.
    const std::vector<const void *> po_rhs
            = binary_injector_utils::prepare_binary_args(post_ops, ctx);
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
