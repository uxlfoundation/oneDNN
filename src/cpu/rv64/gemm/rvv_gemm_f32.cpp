/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
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

#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/rv64/gemm/jit_rvv_gemm_kernel.hpp"
#include "cpu/rv64/gemm/rvv_gemm_f32.hpp"
#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::utils;
using namespace gemm_utils;
using gemm_f32_traits = gemm_utils::gemm_utils_traits<float>;

namespace {

// Scalar copy of A into workspace for cache-friendly access.
// Copies m rows x K columns of A into a contiguous buffer ws.
// After copy, ws is laid out as K blocks of m contiguous elements:
//   ws[k * m + i] = A_logical[i, k]
void copy_A(
        bool isTransA, dim_t K, const float *A, dim_t lda, float *ws, dim_t m) {
    for (dim_t k = 0; k < K; k++) {
        if (isTransA) {
            for (dim_t i = 0; i < m; i++)
                ws[i] = A[i * lda + k];
        } else {
            std::memcpy(ws, A + k * lda, m * sizeof(float));
        }
        ws += m;
    }
}

template <bool isTransA, bool isTransB>
void block_ker(const dim_t M, const dim_t N, const dim_t K, const float *A,
        const dim_t lda, const float *B, const dim_t ldb, float *C,
        const dim_t ldc, const float alpha, const float beta, float *ws,
        bool do_copy, int ithr, const float *bias) {
    MAYBE_UNUSED(ithr);

    const dim_t n_unroll = gemm_f32_traits::get_n_unroll_factor();
    const dim_t m_unroll = gemm_f32_traits::get_m_unroll_factor();

    const dim_t Nu = rnd_dn(N, n_unroll);
    const dim_t Mu = rnd_dn(M, m_unroll);
    const dim_t n_tail = N - Nu;
    const dim_t m_tail = M - Mu;

    auto invoke_kernel
            = [&](const float *a_orig, const float *b, float *c, dim_t tile_m,
                      dim_t tile_n, dim_t j_col, const float *bias_tile) {
        const float *a_eff;
        dim_t lda_eff;
        bool trans_a_eff;

        if (do_copy && tile_m == m_unroll) {
            if (j_col == 0) { copy_A(isTransA, K, a_orig, lda, ws, m_unroll); }
            a_eff = ws;
            lda_eff = m_unroll;
            trans_a_eff = false;
        } else {
            a_eff = a_orig;
            lda_eff = lda;
            trans_a_eff = isTransA;
        }

        jit_rvv_gemm_kernel(a_eff, b, c, lda_eff, ldb, ldc, K, alpha, beta,
                tile_m, tile_n, trans_a_eff, isTransB, bias_tile);
    };

    for (dim_t i = 0; i < Mu; i += m_unroll) {
        const float *a = isTransA ? &A[i * lda] : &A[i];
        const float *bias_tile = bias ? bias + i : nullptr;
        for (dim_t j = 0; j < Nu; j += n_unroll) {
            const float *b = isTransB ? &B[j] : &B[j * ldb];
            invoke_kernel(
                    a, b, &C[i + j * ldc], m_unroll, n_unroll, j, bias_tile);
        }

        if (n_tail > 0) {
            const float *b = isTransB ? &B[Nu] : &B[Nu * ldb];
            invoke_kernel(
                    a, b, &C[i + Nu * ldc], m_unroll, n_tail, Nu, bias_tile);
        }
    }

    if (m_tail > 0) {
        const float *a_tail = isTransA ? &A[Mu * lda] : &A[Mu];
        const float *bias_tile = bias ? bias + Mu : nullptr;

        for (dim_t j = 0; j < Nu; j += n_unroll) {
            const float *b = isTransB ? &B[j] : &B[j * ldb];
            jit_rvv_gemm_kernel(a_tail, b, &C[Mu + j * ldc], lda, ldb, ldc, K,
                    alpha, beta, m_tail, n_unroll, isTransA, isTransB,
                    bias_tile);
        }

        if (n_tail > 0) {
            const float *b = isTransB ? &B[Nu] : &B[Nu * ldb];
            jit_rvv_gemm_kernel(a_tail, b, &C[Mu + Nu * ldc], lda, ldb, ldc, K,
                    alpha, beta, m_tail, n_tail, isTransA, isTransB, bias_tile);
        }
    }
}

template <bool isTransA, bool isTransB>
void gemm_ithr(const dim_t M, const dim_t N, const dim_t K, const float alpha,
        const float *A, const dim_t lda, const float *B, const dim_t ldb,
        const float beta, float *C, const dim_t ldc, bool do_copy, float *ws,
        int ithr, const float *bias) {

    constexpr dim_t BM = gemm_traits_t<float, isTransA, isTransB>::BM;
    constexpr dim_t BN = gemm_traits_t<float, isTransA, isTransB>::BN;
    constexpr dim_t BK = gemm_traits_t<float, isTransA, isTransB>::BK;

    const float *curA;
    const float *curB;
    float *curC;

    if ((M <= 0) || (N <= 0)) return;

    if ((K <= 0) || (alpha == 0.f)) {
        dim_t MN = N * M;
        if (beta == 0.f) {
            for (dim_t j = 0; j < MN; j++)
                C[j] = 0.f;
        } else if (beta != 1.f) {
            for (dim_t j = 0; j < MN; j++)
                C[j] *= beta;
        }
        if (bias) {
            for (dim_t j = 0; j < M; j++)
                for (dim_t i = 0; i < N; i++)
                    C[i * ldc + j] += bias[j];
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
                curB = isTransB ? B + Bn + Bk * ldb : B + Bk + Bn * ldb;
                curC = C + Bm + Bn * ldc;

                if (Bk == 0) {
                    const float *bias_block = bias ? bias + Bm : nullptr;
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, beta, ws, do_copy, ithr,
                            bias_block);
                } else {
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, 1.0f, ws, do_copy, ithr,
                            nullptr);
                }
            }
        }
    }
}
} // namespace

status_t rvv_gemm_f32(const char *transa_, const char *transb_, const dim_t *M_,
        const dim_t *N_, const dim_t *K_, const float *alpha_, const float *A,
        const dim_t *lda_, const float *B, const dim_t *ldb_,
        const float *beta_, float *C, const dim_t *ldc_, const float *bias) {

    if (!(utils::one_of(*transa_, 'n', 'N', 't', 'T')
                && utils::one_of(*transb_, 'n', 'N', 't', 'T')))
        return status::unimplemented;

    bool isTransA = (*transa_ == 'T' || *transa_ == 't');
    bool isTransB = (*transb_ == 'T' || *transb_ == 't');

    const dim_t M = *M_, N = *N_, K = *K_;
    const dim_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const float alpha = *alpha_, beta = *beta_;

    // early out and avoid division by zero in partitioning
    if (utils::one_of(0, M, N)) return status::success;

    int max_nthr = dnnl_get_current_num_threads();
    int nthr_m, nthr_n, nthr_k;
    dim_t MB, NB, KB;

    calc_nthr_nocopy_rvv(
            M, N, K, max_nthr, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(IMPLICATION(!dnnl_thr_syncable(), nthr_k == 1));

    float *c_buffers = nullptr;
    float *ws_buffers = nullptr;
    if (nthr_k > 1) {
        c_buffers = (float *)malloc(
                sizeof(*c_buffers) * nthr_m * nthr_n * (nthr_k - 1) * MB * NB,
                PAGE_4K);
        if (!c_buffers) {
            nthr_k = 1;
            KB = K;
        }
    }

    bool do_copy = (NB / gemm_f32_traits::get_n_unroll_factor() > 3);
    const int nthr_mn = nthr_m * nthr_n;
    const int nthr_to_use = nthr_mn * nthr_k;
    const size_t ws_elems_per_thr = K * gemm_f32_traits::get_m_unroll_factor();
    const size_t ws_size_per_thr
            = rnd_up(ws_elems_per_thr * sizeof(float), PAGE_4K);
    if (do_copy) {
        ws_buffers = (float *)malloc(nthr_to_use * ws_size_per_thr, PAGE_4K);
        if (!ws_buffers) do_copy = false;
    }

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

        float *ws = do_copy
                ? ws_buffers + ithr * ws_size_per_thr / sizeof(float)
                : nullptr;

        dim_t m_from = 0, m_to = 0, myM = 0, n_from = 0, n_to = 0, myN = 0,
              k_from = 0, k_to = 0, myK = 0;

        get_thr_block(m_from, m_to, myM, MB, M, ithr_m);
        get_thr_block(n_from, n_to, myN, NB, N, ithr_n);
        get_thr_block(k_from, k_to, myK, KB, K, ithr_k);

        if (myM > 0 && myN > 0) {
            float myBeta, *myC;
            dim_t ld;
            if (ithr_k == 0) {
                myC = &(C[m_from + n_from * ldc]);
                myBeta = beta;
                ld = ldc;
            } else {
                myC = c_buffers + MB * NB * (cbase + ithr_k - 1);
                myBeta = 0.0f;
                ld = MB;
            }
            const float *myA = isTransA ? &(A[k_from + m_from * lda])
                                        : &(A[m_from + k_from * lda]);
            const float *myB = isTransB ? &(B[n_from + k_from * ldb])
                                        : &(B[k_from + n_from * ldb]);

            const float *myBias
                    = (ithr_k == 0 && bias) ? bias + m_from : nullptr;

            if (!isTransA) {
                if (!isTransB) {
                    gemm_ithr<false, false>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws, ithr, myBias);
                } else {
                    gemm_ithr<false, true>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws, ithr, myBias);
                }
            } else {
                if (!isTransB) {
                    gemm_ithr<true, false>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws, ithr, myBias);
                } else {
                    gemm_ithr<true, true>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws, ithr, myBias);
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
                float *myC = c_buffers
                        + MB * ((dim_t)NB * (cbase + ik - 1) + offset);

                gemm_utils::sum_two_matrices(myM, block, myC, MB,
                        &C[m_from + (n_from + offset) * ldc], ldc);
            }
        });
    }

    free(ws_buffers);
    free(c_buffers);

    return status::success;
}
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
