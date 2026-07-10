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
#include "common/nstl.hpp"
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
        bool do_copy, int ithr, const float *bias, const dim_t m_unroll,
        bool b_signed, bool dst_is_f32,
        const jit_rvv_gemm_s8_kernel_table_t &trans_a_table,
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
        const float *bias_tile = bias ? bias + i : nullptr;
        for (dim_t j = 0; j < Nu; j += n_unroll) {
            const char *b = isTransB ? static_cast<const char *>(B) + j
                                     : static_cast<const char *>(B) + j * ldb;
            invoke_kernel(a, b, static_cast<char *>(C) + (i + j * ldc) * 4,
                    m_unroll, n_unroll, j, bias_tile);
        }

        if (n_tail > 0) {
            const char *b = isTransB ? static_cast<const char *>(B) + Nu
                                     : static_cast<const char *>(B) + Nu * ldb;
            invoke_kernel(a, b, static_cast<char *>(C) + (i + Nu * ldc) * 4,
                    m_unroll, n_tail, Nu, bias_tile);
        }
    }

    if (m_tail > 0) {
        const int8_t *a_tail = isTransA ? &A[Mu * lda] : &A[Mu];
        const float *bias_tile = bias ? bias + Mu : nullptr;

        for (dim_t j = 0; j < Nu; j += n_unroll) {
            const char *b = isTransB ? static_cast<const char *>(B) + j
                                     : static_cast<const char *>(B) + j * ldb;
            const auto &kernel_table
                    = isTransA ? trans_a_table : nontrans_a_table;
            call_kernel(kernel_table, a_tail, b,
                    static_cast<char *>(C) + (Mu + j * ldc) * 4, lda, m_tail,
                    n_unroll, bias_tile);
        }

        if (n_tail > 0) {
            const char *b = isTransB ? static_cast<const char *>(B) + Nu
                                     : static_cast<const char *>(B) + Nu * ldb;
            const auto &kernel_table
                    = isTransA ? trans_a_table : nontrans_a_table;
            call_kernel(kernel_table, a_tail, b,
                    static_cast<char *>(C) + (Mu + Nu * ldc) * 4, lda, m_tail,
                    n_tail, bias_tile);
        }
    }
}

template <bool isTransA, bool isTransB>
void gemm_ithr_s8(const dim_t M, const dim_t N, const dim_t K,
        const float alpha, const int8_t *A, const dim_t lda, const void *B,
        const dim_t ldb, const float beta, void *C, const dim_t ldc,
        bool do_copy, int8_t *ws, int ithr, const float *bias,
        const dim_t m_unroll, bool b_signed, bool dst_is_f32,
        const jit_rvv_gemm_s8_kernel_table_t &trans_a_table,
        const jit_rvv_gemm_s8_kernel_table_t &nontrans_a_table) {

    constexpr dim_t BM = gemm_traits_t<float, isTransA, isTransB>::BM;
    constexpr dim_t BN = gemm_traits_t<float, isTransA, isTransB>::BN;
    constexpr dim_t BK = gemm_traits_t<float, isTransA, isTransB>::BK;

    const int8_t *curA;
    const void *curB;
    char *curC;

    if ((M <= 0) || (N <= 0)) return;

    if ((K <= 0) || (alpha == 0.f)) {
        dim_t MN = N * M;
        if (dst_is_f32) {
            if (beta == 0.f) {
                std::memset(C, 0, sizeof(float) * MN);
            } else if (beta != 1.f) {
                float *C_f = static_cast<float *>(C);
                for (dim_t j = 0; j < MN; j++)
                    C_f[j] *= beta;
            }
            if (bias) {
                float *C_f = static_cast<float *>(C);
                for (dim_t j = 0; j < M; j++)
                    for (dim_t i = 0; i < N; i++)
                        C_f[i * ldc + j] += bias[j];
            }
        } else {
            // s32 dst, alpha == 0: result = beta*C + s32(bias).
            int32_t *C_i = static_cast<int32_t *>(C);
            if (beta == 0.f) {
                std::memset(C_i, 0, sizeof(int32_t) * MN);
            } else if (beta != 1.f) {
                for (dim_t j = 0; j < MN; j++)
                    C_i[j] = static_cast<int32_t>(beta * C_i[j]);
            }
            if (bias) {
                for (dim_t j = 0; j < M; j++)
                    for (dim_t i = 0; i < N; i++)
                        C_i[i * ldc + j] += static_cast<int32_t>(bias[j]);
            }
        }
        return;
    }

    for (dim_t Bk = 0; Bk < K; Bk += BK) {
        dim_t kb = nstl::min(K - Bk, BK);
        for (dim_t Bm = 0; Bm < M; Bm += BM) {
            dim_t mb = nstl::min(M - Bm, BM);
            for (dim_t Bn = 0; Bn < N; Bn += BN) {
                dim_t nb = nstl::min(N - Bn, BN);
                curA = isTransA ? A + Bk + Bm * lda : A + Bm + Bk * lda;
                const char *B_bytes = static_cast<const char *>(B);
                curB = isTransB
                        ? static_cast<const void *>(B_bytes + Bn + Bk * ldb)
                        : static_cast<const void *>(B_bytes + Bk + Bn * ldb);
                curC = static_cast<char *>(C) + (Bm + Bn * ldc) * 4;

                if (Bk == 0) {
                    const float *bias_block = bias ? bias + Bm : nullptr;
                    block_ker_s8<isTransA, isTransB>(mb, nb, kb, curA, lda,
                            curB, ldb, curC, ldc, alpha, beta, ws, do_copy,
                            ithr, bias_block, m_unroll, b_signed, dst_is_f32,
                            trans_a_table, nontrans_a_table);
                } else {
                    block_ker_s8<isTransA, isTransB>(mb, nb, kb, curA, lda,
                            curB, ldb, curC, ldc, alpha, 1.0f, ws, do_copy,
                            ithr, nullptr, m_unroll, b_signed, dst_is_f32,
                            trans_a_table, nontrans_a_table);
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
        bool b_signed, bool dst_is_f32, int32_t *c_buffers_in,
        int8_t *ws_buffers_in) {

    if (!(utils::one_of(*transa_, 'n', 'N', 't', 'T')
                && utils::one_of(*transb_, 'n', 'N', 't', 'T')))
        return status::unimplemented;

    bool isTransA = (*transa_ == 'T' || *transa_ == 't');
    bool isTransB = (*transb_ == 'T' || *transb_ == 't');

    const dim_t M = *M_, N = *N_, K = *K_;
    const dim_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const float alpha = *alpha_, beta = *beta_;

    // Early out: avoid division by zero in partitioning.
    if (utils::one_of(0, M, N)) return status::success;

    // Use current (not max) threads
    int nthr = dnnl_get_current_num_threads();
    int nthr_m, nthr_n, nthr_k;
    dim_t MB, NB, KB;

    calc_nthr_nocopy_rvv(
            M, N, K, nthr, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(IMPLICATION(!dnnl_thr_syncable(), nthr_k == 1));

    // Copy A into a cache-friendly contiguous workspace once per M-tile when
    // there are enough N-tiles per thread (>= 4 full tiles)
    bool do_copy = (NB / gemm_s8_traits::get_n_unroll_factor() > 3);
    const int nthr_mn = nthr_m * nthr_n;
    const int nthr_to_use = nthr_mn * nthr_k;
    const dim_t m_unroll = gemm_s8_traits::get_m_unroll_factor();
    const size_t ws_elems_per_thr = K * m_unroll;
    const size_t ws_size_per_thr
            = rnd_up(ws_elems_per_thr * sizeof(int8_t), PAGE_4K);

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
    if (do_copy && !ws_buffers) {
        ws_buffers = static_cast<int8_t *>(
                malloc(nthr_to_use * ws_size_per_thr, PAGE_4K));
        own_ws = ws_buffers != nullptr;
        if (!own_ws) do_copy = false;
    }

    const auto &trans_a_table = get_jit_rvv_gemm_s8_kernel_table(
            isTransA, isTransB, b_signed, dst_is_f32);
    const auto &nontrans_a_table = get_jit_rvv_gemm_s8_kernel_table(
            false, isTransB, b_signed, dst_is_f32);

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

        if (myM > 0 && myN > 0) {
            void *myC;
            float myBeta;
            dim_t ld;
            if (ithr_k == 0) {
                myC = static_cast<char *>(C) + (m_from + n_from * ldc) * 4;
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

            const float *myBias
                    = (ithr_k == 0 && bias) ? bias + m_from : nullptr;

            if (!isTransA) {
                if (!isTransB) {
                    gemm_ithr_s8<false, false>(myM, myN, myK, alpha, myA, lda,
                            myB, ldb, myBeta, myC, ld, do_copy, ws, ithr,
                            myBias, m_unroll, b_signed, dst_is_f32,
                            trans_a_table, nontrans_a_table);
                } else {
                    gemm_ithr_s8<false, true>(myM, myN, myK, alpha, myA, lda,
                            myB, ldb, myBeta, myC, ld, do_copy, ws, ithr,
                            myBias, m_unroll, b_signed, dst_is_f32,
                            trans_a_table, nontrans_a_table);
                }
            } else {
                if (!isTransB) {
                    gemm_ithr_s8<true, false>(myM, myN, myK, alpha, myA, lda,
                            myB, ldb, myBeta, myC, ld, do_copy, ws, ithr,
                            myBias, m_unroll, b_signed, dst_is_f32,
                            trans_a_table, nontrans_a_table);
                } else {
                    gemm_ithr_s8<true, true>(myM, myN, myK, alpha, myA, lda,
                            myB, ldb, myBeta, myC, ld, do_copy, ws, ithr,
                            myBias, m_unroll, b_signed, dst_is_f32,
                            trans_a_table, nontrans_a_table);
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

            dim_t offset = 0, block = 0;

            gemm_utils::partition_unit_diff(
                    ithr_k, nthr_k, myN, &offset, &block);
            for (int ik = 1; ik < nthr_k; ++ik) {
                if (dst_is_f32) {
                    float *myC = reinterpret_cast<float *>(c_buffers)
                            + MB * ((dim_t)NB * (cbase + ik - 1) + offset);
                    gemm_utils::sum_two_matrices(myM, block, myC, MB,
                            static_cast<float *>(C) + m_from
                                    + (n_from + offset) * ldc,
                            ldc);
                } else {
                    int32_t *myC = c_buffers
                            + MB * ((dim_t)NB * (cbase + ik - 1) + offset);
                    gemm_utils::sum_two_matrices(myM, block, myC, MB,
                            static_cast<int32_t *>(C) + m_from
                                    + (n_from + offset) * ldc,
                            ldc);
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
