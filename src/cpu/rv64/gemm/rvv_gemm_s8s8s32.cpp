/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/rv64/gemm/jit_rvv_gemm_kernel.hpp"
#include "cpu/rv64/gemm/jit_rvv_gemm_s8_kernel.hpp"
#include "cpu/rv64/gemm/rvv_gemm_s8s8s32.hpp"
#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::utils;
using namespace gemm_utils;
using gemm_s8_traits = gemm_utils::gemm_utils_traits<int8_t>;

namespace {

static bool dst_is_overwrite_only(data_type_t dt) {
    return one_of(
            dt, data_type::s8, data_type::u8, data_type::f16, data_type::bf16);
}

static bool dst_supports_k_split(data_type_t dt) {
    return one_of(dt, data_type::s32, data_type::f32);
}

// Scalar copy of A (s8 weights) into a cache-friendly workspace. After copy,
// ws holds K blocks of m_unroll contiguous s8 values:
//   ws[k * m_unroll + i] = A_logical[i, k]
void copy_A_s8(bool isTransA, dim_t K, const int8_t *A, dim_t lda, int8_t *ws,
        dim_t m) {
    for (dim_t k = 0; k < K; k++) {
        if (isTransA) {
            for (dim_t i = 0; i < m; i++)
                ws[i] = A[i * lda + k];
        } else {
            std::memcpy(ws, A + k * lda, m * sizeof(int8_t));
        }
        ws += m;
    }
}

template <bool isTransA, bool isTransB>
void block_ker_s8(const dim_t M, const dim_t N, const dim_t K, const int8_t *A,
        const dim_t lda, const void *B, const dim_t ldb, void *C,
        const dim_t ldc, const float alpha, const float beta, int8_t *ws,
        bool do_copy, int ithr, const float *bias, bool bias_is_scalar,
        const dim_t m_unroll, bool b_signed, data_type_t dst_dt,
        size_t dst_dt_sz, const jit_rvv_gemm_s8_kernel_table_t &trans_a_table,
        const jit_rvv_gemm_s8_kernel_table_t &nontrans_a_table) {
    MAYBE_UNUSED(ithr);

    const dim_t n_unroll = gemm_s8_traits::get_n_unroll_factor();

    const dim_t Nu = rnd_dn(N, n_unroll);
    const dim_t Mu = rnd_dn(M, m_unroll);
    const dim_t n_tail = N - Nu;
    const dim_t m_tail = M - Mu;

    auto call_kernel
            = [&](const jit_rvv_gemm_s8_kernel_table_t &kernel_table,
                      const void *a, const void *b, void *c, dim_t lda_eff,
                      dim_t tile_m, dim_t tile_n, const float *bias_tile) {
        jit_rvv_gemm_s8_kernel_t::call_params_t p;
        p.A = a;
        p.B = b;
        p.C = c;
        p.lda = lda_eff;
        p.ldb = ldb;
        p.ldc = ldc;
        p.K = K;
        p.m = tile_m;
        p.alpha = alpha;
        p.beta = beta;
        p.bias = bias_tile;
        p.bias_is_scalar = bias_is_scalar;

        const jit_rvv_gemm_s8_kernel_t *kernel
                = bias_tile ? kernel_table.b[tile_n] : kernel_table.nb[tile_n];
        (*kernel)(&p);
    };

    auto invoke_kernel
            = [&](const int8_t *a_orig, const void *b, void *c, dim_t tile_m,
                      dim_t tile_n, dim_t j_col, const float *bias_tile) {
        const void *a_eff;
        dim_t lda_eff;
        bool trans_a_eff;

        if (do_copy && tile_m == m_unroll) {
            if (j_col == 0) {
                copy_A_s8(isTransA, K, a_orig, lda, ws, m_unroll);
            }
            a_eff = ws;
            lda_eff = m_unroll;
            trans_a_eff = false;
        } else {
            a_eff = a_orig;
            lda_eff = lda;
            trans_a_eff = isTransA;
        }

        const auto &kernel_table
                = trans_a_eff ? trans_a_table : nontrans_a_table;
        call_kernel(
                kernel_table, a_eff, b, c, lda_eff, tile_m, tile_n, bias_tile);
    };

    for (dim_t i = 0; i < Mu; i += m_unroll) {
        const int8_t *a = isTransA ? &A[i * lda] : &A[i];
        // Scalar bias broadcasts over M: every tile reads the same base float.
        const float *bias_tile
                = bias ? (bias_is_scalar ? bias : bias + i) : nullptr;
        for (dim_t j = 0; j < Nu; j += n_unroll) {
            const char *b = isTransB ? static_cast<const char *>(B) + j
                                     : static_cast<const char *>(B) + j * ldb;
            invoke_kernel(a, b,
                    static_cast<char *>(C) + (i + j * ldc) * dst_dt_sz,
                    m_unroll, n_unroll, j, bias_tile);
        }

        if (n_tail > 0) {
            const char *b = isTransB ? static_cast<const char *>(B) + Nu
                                     : static_cast<const char *>(B) + Nu * ldb;
            invoke_kernel(a, b,
                    static_cast<char *>(C) + (i + Nu * ldc) * dst_dt_sz,
                    m_unroll, n_tail, Nu, bias_tile);
        }
    }

    if (m_tail > 0) {
        const int8_t *a_tail = isTransA ? &A[Mu * lda] : &A[Mu];
        const float *bias_tile
                = bias ? (bias_is_scalar ? bias : bias + Mu) : nullptr;

        for (dim_t j = 0; j < Nu; j += n_unroll) {
            const char *b = isTransB ? static_cast<const char *>(B) + j
                                     : static_cast<const char *>(B) + j * ldb;
            const auto &kernel_table
                    = isTransA ? trans_a_table : nontrans_a_table;
            call_kernel(kernel_table, a_tail, b,
                    static_cast<char *>(C) + (Mu + j * ldc) * dst_dt_sz, lda,
                    m_tail, n_unroll, bias_tile);
        }

        if (n_tail > 0) {
            const char *b = isTransB ? static_cast<const char *>(B) + Nu
                                     : static_cast<const char *>(B) + Nu * ldb;
            const auto &kernel_table
                    = isTransA ? trans_a_table : nontrans_a_table;
            call_kernel(kernel_table, a_tail, b,
                    static_cast<char *>(C) + (Mu + Nu * ldc) * dst_dt_sz, lda,
                    m_tail, n_tail, bias_tile);
        }
    }
}

template <bool isTransA, bool isTransB>
void gemm_ithr_s8(const dim_t M, const dim_t N, const dim_t K,
        const float alpha, const int8_t *A, const dim_t lda, const void *B,
        const dim_t ldb, const float beta, void *C, const dim_t ldc,
        bool do_copy, int8_t *ws, int ithr, const float *bias,
        bool bias_is_scalar, const dim_t m_unroll, bool b_signed,
        data_type_t dst_dt, size_t dst_dt_sz,
        const jit_rvv_gemm_s8_kernel_table_t &trans_a_table,
        const jit_rvv_gemm_s8_kernel_table_t &nontrans_a_table) {

    constexpr dim_t BM = gemm_traits_t<float, isTransA, isTransB>::BM;
    constexpr dim_t BN = gemm_traits_t<float, isTransA, isTransB>::BN;

    const dim_t BK_eff = dst_is_overwrite_only(dst_dt)
            ? K
            : gemm_traits_t<float, isTransA, isTransB>::BK;

    const int8_t *curA;
    const void *curB;
    char *curC;

    if ((M <= 0) || (N <= 0)) return;

    if (K <= 0) {
        auto scalar_bias = [&](dim_t j) {
            return bias ? (bias_is_scalar ? bias[0] : bias[j]) : 0.f;
        };
        const dim_t MN = N * M;
        if (dst_is_overwrite_only(dst_dt)) {
            for (dim_t j = 0; j < M; j++) {
                const float v = scalar_bias(j);
                for (dim_t i = 0; i < N; i++) {
                    char *d = static_cast<char *>(C)
                            + (j + i * ldc) * dst_dt_sz;
                    if (dst_dt == data_type::s8) {
                        const int r
                                = math::mxcsr_cvt(saturate(-128.f, 127.f, v));
                        *d = static_cast<char>(static_cast<int8_t>(r));
                    } else if (dst_dt == data_type::u8) {
                        const int r = math::mxcsr_cvt(saturate(0.f, 255.f, v));
                        *d = static_cast<char>(static_cast<uint8_t>(r));
                    } else {
                        types::cvt_from_float(dst_dt, d, &v, 1);
                    }
                }
            }
        } else if (dst_dt == data_type::f32) {
            float *C_f = static_cast<float *>(C);
            if (beta == 0.f)
                std::memset(C_f, 0, sizeof(float) * MN);
            else if (beta != 1.f)
                for (dim_t j = 0; j < MN; j++)
                    C_f[j] *= beta;
            if (bias)
                for (dim_t j = 0; j < M; j++) {
                    const float v = scalar_bias(j);
                    for (dim_t i = 0; i < N; i++)
                        C_f[i * ldc + j] += v;
                }
        } else {
            int32_t *C_i = static_cast<int32_t *>(C);
            if (beta == 0.f)
                std::memset(C_i, 0, sizeof(int32_t) * MN);
            else if (beta != 1.f)
                for (dim_t j = 0; j < MN; j++)
                    C_i[j] = static_cast<int32_t>(beta * C_i[j]);
            if (bias)
                for (dim_t j = 0; j < M; j++) {
                    const int32_t bv = static_cast<int32_t>(
                            math::mxcsr_cvt(scalar_bias(j)));
                    for (dim_t i = 0; i < N; i++)
                        C_i[i * ldc + j] += bv;
                }
        }
        return;
    }

    for (dim_t Bk = 0; Bk < K; Bk += BK_eff) {
        dim_t kb = nstl::min(K - Bk, BK_eff);
        for (dim_t Bm = 0; Bm < M; Bm += BM) {
            dim_t mb = nstl::min(M - Bm, BM);
            for (dim_t Bn = 0; Bn < N; Bn += BN) {
                dim_t nb = nstl::min(N - Bn, BN);
                curA = isTransA ? A + Bk + Bm * lda : A + Bm + Bk * lda;
                const char *B_bytes = static_cast<const char *>(B);
                curB = isTransB
                        ? static_cast<const void *>(B_bytes + Bn + Bk * ldb)
                        : static_cast<const void *>(B_bytes + Bk + Bn * ldb);
                curC = static_cast<char *>(C) + (Bm + Bn * ldc) * dst_dt_sz;

                if (Bk == 0) {
                    const float *bias_block = bias
                            ? (bias_is_scalar ? bias : bias + Bm)
                            : nullptr;
                    block_ker_s8<isTransA, isTransB>(mb, nb, kb, curA, lda,
                            curB, ldb, curC, ldc, alpha, beta, ws, do_copy,
                            ithr, bias_block, bias_is_scalar, m_unroll,
                            b_signed, dst_dt, dst_dt_sz, trans_a_table,
                            nontrans_a_table);
                } else {
                    block_ker_s8<isTransA, isTransB>(mb, nb, kb, curA, lda,
                            curB, ldb, curC, ldc, alpha, 1.0f, ws, do_copy,
                            ithr, nullptr, bias_is_scalar, m_unroll, b_signed,
                            dst_dt, dst_dt_sz, trans_a_table, nontrans_a_table);
                }
            }
        }
    }
}
} // namespace

status_t rvv_gemm_s8s8s32(const char *transa_, const char *transb_,
        const dim_t *M_, const dim_t *N_, const dim_t *K_, const float *alpha_,
        const int8_t *A, const dim_t *lda_, const void *B, const dim_t *ldb_,
        const float *beta_, void *C, const dim_t *ldc_, const float *bias,
        bool b_signed, data_type_t dst_dt, int32_t *c_buffers_in,
        int8_t *ws_buffers_in, bool bias_is_scalar,
        const gemm_partition_t *part) {

    if (!(utils::one_of(*transa_, 'n', 'N', 't', 'T')
                && utils::one_of(*transb_, 'n', 'N', 't', 'T')))
        return status::unimplemented;

    bool isTransA = (*transa_ == 'T' || *transa_ == 't');
    bool isTransB = (*transb_ == 'T' || *transb_ == 't');

    const dim_t M = *M_, N = *N_, K = *K_;
    const dim_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const float alpha = *alpha_, beta = *beta_;

    // Per-dt alpha/beta contracts (mirroring the existing s32 path's reject
    // semantics; extended to all narrow paths):
    //   - s32  : alpha must be 1;  beta in {0,1}
    //   - f32  : any alpha/beta  (full epilogue)
    //   - f16  : any alpha; beta must be 0 (overwrite-only epilogue)
    //   - bf16 : any alpha; beta must be 0 (overwrite-only epilogue)
    //   - s8   : alpha ignored (asserted == 1); beta must be 0 (overwrite)
    //   - u8   : alpha ignored (asserted == 1); beta must be 0 (overwrite)
    if (dst_dt == data_type::s32) {
        if (alpha != 1.0f || !utils::one_of(beta, 0.0f, 1.0f))
            return status::unimplemented;
    } else if (one_of(dst_dt, data_type::s8, data_type::u8)) {
        if (alpha != 1.0f || beta != 0.0f) return status::unimplemented;
    } else if (one_of(dst_dt, data_type::f16, data_type::bf16)) {
        // f16/bf16 epilogue loads one half-width scalar and broadcasts it
        // into the m-lane vector via vfmv.v.f. This is correct only when
        // beta==0 (overwrite); for beta!=0 the per-lane C[i] values get
        // clobbered to C[0]. Reject to keep the contract explicit rather
        // than silently producing wrong results for any future caller that
        // passes beta!=0 with half-precision dst.
        if (beta != 0.0f) return status::unimplemented;
    }
    // f32 accepts any alpha/beta (the kernel implements the epilogue).

    // Early out: avoid division by zero in partitioning.
    if (utils::one_of(0, M, N)) return status::success;

    int nthr_m, nthr_n, nthr_k;
    dim_t MB, NB, KB;
    if (part) {
        nthr_m = part->nthr_m;
        nthr_n = part->nthr_n;
        nthr_k = part->nthr_k;
        MB = part->MB;
        NB = part->NB;
        KB = part->KB;
    } else {
        int nthr = dnnl_get_current_num_threads();
        calc_nthr_nocopy_rvv(
                M, N, K, nthr, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    }
    assert(IMPLICATION(!dnnl_thr_syncable(), nthr_k == 1));

    bool do_copy = (NB / gemm_s8_traits::get_n_unroll_factor() > 3);
    const int nthr_mn = nthr_m * nthr_n;
    const dim_t m_unroll = gemm_s8_traits::get_m_unroll_factor();
    const size_t ws_elems_per_thr = K * m_unroll;
    const size_t ws_size_per_thr
            = rnd_up(ws_elems_per_thr * sizeof(int8_t), PAGE_4K);

    assert(utils::one_of(dst_dt, data_type::s32, data_type::f32, data_type::s8,
            data_type::u8, data_type::f16, data_type::bf16));
    const size_t dst_dt_sz = types::data_type_size(dst_dt);

    // K-split is only supported for dst types whose JIT epilogue implements
    // the read-modify-write contract (s32 and f32 — see dst_supports_k_split).
    // For s8/u8/f16/bf16 the overwrite-only narrow epilogue would silently
    // drop the prior K-block's contribution on the K-split path (Bk > 0 is
    // invoked with beta=1).
    const bool supports_k_split = dst_supports_k_split(dst_dt);
    if (!supports_k_split) {
        nthr_k = 1;
        KB = K;
    }

    int32_t *c_buffers = c_buffers_in;
    int8_t *ws_buffers = ws_buffers_in;
    bool own_c = false, own_ws = false;
    if (nthr_k > 1 && !c_buffers) {
        c_buffers = static_cast<int32_t *>(malloc(
                sizeof(*c_buffers) * nthr_m * nthr_n * (nthr_k - 1) * MB * NB,
                PAGE_4K));
        own_c = c_buffers != nullptr;
        if (!own_c) {
            nthr_k = 1;
            KB = K;
        }
    }
    const int nthr_to_use = nthr_mn * nthr_k;
    if (do_copy && !ws_buffers) {
        ws_buffers = static_cast<int8_t *>(
                malloc(nthr_to_use * ws_size_per_thr, PAGE_4K));
        own_ws = ws_buffers != nullptr;
        if (!own_ws) do_copy = false;
    }

    // Cache the per-(transA, dst_dt) table outside the parallel so the
    // inner work fns reuse a single 12-pointer struct rather than
    // rebuilding it per tile.
    const auto trans_a_table = get_jit_rvv_gemm_s8_kernel_table(
            isTransA, isTransB, b_signed, dst_dt);
    const auto nontrans_a_table = get_jit_rvv_gemm_s8_kernel_table(
            false, isTransB, b_signed, dst_dt);

    auto get_thr_block = [&](dim_t &from, dim_t &to, dim_t &myN, dim_t NB,
                                 dim_t N, int ithr) {
        from = NB * (ithr);
        to = NB * (ithr + 1);
        if (to > N) to = N;
        myN = to - from;
    };

    parallel(nthr_to_use, [&](int ithr, int nthr) {
        assert(nthr_to_use == nthr);
        MAYBE_UNUSED(nthr);

        int ithr_mn = ithr % nthr_mn;
        int ithr_m = ithr_mn % nthr_m;
        int ithr_n = ithr_mn / nthr_m;
        int ithr_k = ithr / nthr_mn;

        int cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

        int8_t *ws = do_copy ? ws_buffers + ithr * ws_size_per_thr : nullptr;

        dim_t m_from = 0, m_to = 0, myM = 0, n_from = 0, n_to = 0, myN = 0,
              k_from = 0, k_to = 0, myK = 0;

        get_thr_block(m_from, m_to, myM, MB, M, ithr_m);
        get_thr_block(n_from, n_to, myN, NB, N, ithr_n);
        get_thr_block(k_from, k_to, myK, KB, K, ithr_k);

        if (myM > 0 && myN > 0 && myK > 0) {
            void *myC;
            float myBeta;
            dim_t ld;
            if (ithr_k == 0) {
                myC = static_cast<char *>(C)
                        + (m_from + n_from * ldc) * dst_dt_sz;
                myBeta = beta;
                ld = ldc;
            } else {
                myC = c_buffers + MB * NB * (cbase + ithr_k - 1);
                myBeta = 0.0f;
                ld = MB;
            }
            const int8_t *myA = isTransA ? &(A[k_from + m_from * lda])
                                         : &(A[m_from + k_from * lda]);
            const char *B_bytes = static_cast<const char *>(B);
            const void *myB = isTransB
                    ? static_cast<const void *>(B_bytes + n_from + k_from * ldb)
                    : static_cast<const void *>(
                              B_bytes + k_from + n_from * ldb);

            const float *myBias = (ithr_k == 0 && bias)
                    ? (bias_is_scalar ? bias : bias + m_from)
                    : nullptr;

            if (!isTransA) {
                if (!isTransB) {
                    gemm_ithr_s8<false, false>(myM, myN, myK, alpha, myA, lda,
                            myB, ldb, myBeta, myC, ld, do_copy, ws, ithr,
                            myBias, bias_is_scalar, m_unroll, b_signed, dst_dt,
                            dst_dt_sz, trans_a_table, nontrans_a_table);
                } else {
                    gemm_ithr_s8<false, true>(myM, myN, myK, alpha, myA, lda,
                            myB, ldb, myBeta, myC, ld, do_copy, ws, ithr,
                            myBias, bias_is_scalar, m_unroll, b_signed, dst_dt,
                            dst_dt_sz, trans_a_table, nontrans_a_table);
                }
            } else {
                if (!isTransB) {
                    gemm_ithr_s8<true, false>(myM, myN, myK, alpha, myA, lda,
                            myB, ldb, myBeta, myC, ld, do_copy, ws, ithr,
                            myBias, bias_is_scalar, m_unroll, b_signed, dst_dt,
                            dst_dt_sz, trans_a_table, nontrans_a_table);
                } else {
                    gemm_ithr_s8<true, true>(myM, myN, myK, alpha, myA, lda,
                            myB, ldb, myBeta, myC, ld, do_copy, ws, ithr,
                            myBias, bias_is_scalar, m_unroll, b_signed, dst_dt,
                            dst_dt_sz, trans_a_table, nontrans_a_table);
                }
            }
        }
    });

    if (nthr_k > 1) {
        parallel(nthr_to_use, [&](int ithr, int nthr) {
            assert(nthr_to_use == nthr);
            MAYBE_UNUSED(nthr);

            int ithr_mn = ithr % nthr_mn;
            int ithr_m = ithr_mn % nthr_m;
            int ithr_k = ithr / nthr_mn;
            int ithr_n = ithr_mn / nthr_m;

            dim_t n_from = 0, n_to = 0, myN = 0;
            dim_t m_from = 0, m_to = 0, myM = 0;

            int cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

            get_thr_block(n_from, n_to, myN, NB, N, ithr_n);
            get_thr_block(m_from, m_to, myM, MB, M, ithr_m);
            const bool reduction_tile_empty = !(myM > 0 && myN > 0);
            if (reduction_tile_empty) return;

            dim_t offset = 0, block = 0;

            gemm_utils::partition_unit_diff(
                    ithr_k, nthr_k, myN, &offset, &block);
            for (int ik = 1; ik < nthr_k; ++ik) {
                char *dstC = static_cast<char *>(C)
                        + (m_from + (n_from + offset) * ldc) * dst_dt_sz;
                if (dst_dt == data_type::f32) {
                    float *myC = reinterpret_cast<float *>(c_buffers)
                            + MB * ((dim_t)NB * (cbase + ik - 1) + offset);
                    gemm_utils::sum_two_matrices(myM, block, myC, MB,
                            reinterpret_cast<float *>(dstC), ldc);
                } else if (dst_dt == data_type::s32) {
                    int32_t *myC = c_buffers
                            + MB * ((dim_t)NB * (cbase + ik - 1) + offset);
                    gemm_utils::sum_two_matrices(myM, block, myC, MB,
                            reinterpret_cast<int32_t *>(dstC), ldc);
                } else {
                    // s8/u8/f16/bf16: unreachable (nthr_k forced to 1 by
                    // dst_supports_k_split returning false above).
                    assert(!"s8/u8/f16/bf16 dst must have nthr_k==1 "
                            "(supports_k_split should be false)");
                }
            }
        });
    }

    if (own_ws) free(ws_buffers);
    if (own_c) free(c_buffers);

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
