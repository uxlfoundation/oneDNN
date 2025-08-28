/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/rv64/gemm/rvv_gemm_f32.hpp"
#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"

#include "cpu/gemm/f32/ref_gemm_f32.hpp"

#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::utils;
using namespace gemm_utils;

namespace {
void copy_A(
        bool isTransA, dim_t K, const float *A, const dim_t lda, float *ws) {
    constexpr dim_t m = unroll_factor<float>::m;

    for (dim_t k = 0; k < K; k++) {
        dim_t i = 0;
        if (isTransA) {
            ptrdiff_t stride = lda * sizeof(float);
            while (i < m) {
                size_t vl = __riscv_vsetvl_e32m1(m - i);
                const float *a_ptr = A + i * lda + k;
                vfloat32m1_t v_a = __riscv_vlse32_v_f32m1(a_ptr, stride, vl);
                __riscv_vse32_v_f32m1(ws + i, v_a, vl);
                i += vl;
            }
        } else {
            const float *a_ptr = A + k * lda;
            while (i < m) {
                size_t vl = __riscv_vsetvl_e32m1(m - i);
                vfloat32m1_t v_a = __riscv_vle32_v_f32m1(a_ptr + i, vl);
                __riscv_vse32_v_f32m1(ws + i, v_a, vl);
                i += vl;
            }
        }
        ws += m;
    }
}

template <bool isTransA, bool isTransB>
void kernel_mxn(dim_t K, const float *A, const dim_t lda, const float *B,
        const dim_t ldb, float *C, const dim_t ldc, const float alpha,
        const float beta, int ithr = -1) {
    constexpr dim_t m = unroll_factor<float>::m;
    constexpr dim_t n = unroll_factor<float>::n;

    static_assert(n == 4, "This kernel is specialized for n=4");

    dim_t i = 0;
    while (i < m) {
        size_t vl = __riscv_vsetvl_e32m1(m - i);

        vfloat32m1_t v_c0, v_c1, v_c2, v_c3;

        v_c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (dim_t k = 0; k < K; ++k) {
            vfloat32m1_t v_a;
            if (isTransA) {
                ptrdiff_t stride_a = lda * sizeof(float);
                v_a = __riscv_vlse32_v_f32m1(A + i * lda + k, stride_a, vl);
            } else {
                v_a = __riscv_vle32_v_f32m1(A + i + k * lda, vl);
            }

            const float *b_ptr = isTransB ? &B[k * ldb] : &B[k];
            const dim_t b_stride = isTransB ? 1 : ldb;

            v_c0 = __riscv_vfmacc_vf_f32m1(v_c0, b_ptr[0 * b_stride], v_a, vl);
            v_c1 = __riscv_vfmacc_vf_f32m1(v_c1, b_ptr[1 * b_stride], v_a, vl);
            v_c2 = __riscv_vfmacc_vf_f32m1(v_c2, b_ptr[2 * b_stride], v_a, vl);
            v_c3 = __riscv_vfmacc_vf_f32m1(v_c3, b_ptr[3 * b_stride], v_a, vl);
        }

#define STORE_C(J, V_C) \
    do { \
        float *c_final_ptr = C + (J)*ldc + i; \
        if (beta == 0.0f) { \
            vfloat32m1_t v_res = __riscv_vfmul_vf_f32m1(V_C, alpha, vl); \
            __riscv_vse32_v_f32m1(c_final_ptr, v_res, vl); \
        } else { \
            vfloat32m1_t v_c_old = __riscv_vle32_v_f32m1(c_final_ptr, vl); \
            vfloat32m1_t v_res = __riscv_vfmul_vf_f32m1(v_c_old, beta, vl); \
            v_res = __riscv_vfmacc_vf_f32m1(v_res, alpha, V_C, vl); \
            __riscv_vse32_v_f32m1(c_final_ptr, v_res, vl); \
        } \
    } while (0)

        STORE_C(0, v_c0);
        STORE_C(1, v_c1);
        STORE_C(2, v_c2);
        STORE_C(3, v_c3);

#undef STORE_C
        i += vl;
    }
}

template <bool isTransA, bool isTransB>
void block_ker(const dim_t M, const dim_t N, const dim_t K, const float *A,
        const dim_t lda, const float *B, const dim_t ldb, float *C,
        const dim_t ldc, const float alpha, const float beta, float *ws,
        bool do_copy, int ithr = -1) {

    dim_t Nu = rnd_dn(N, unroll_factor<float>::n);
    dim_t Mu = rnd_dn(M, unroll_factor<float>::m);
    for (dim_t i = 0; i < Mu; i += unroll_factor<float>::m) {
        for (dim_t j = 0; j < Nu; j += unroll_factor<float>::n) {
            const float *b = isTransB ? &B[j] : &B[j * ldb];
            const float *a = isTransA ? &A[i * lda] : &A[i];
            if (do_copy) {
                if (j == 0) { copy_A(isTransA, K, a, lda, ws); }
                kernel_mxn<false, isTransB>(K, ws, unroll_factor<float>::m, b,
                        ldb, &C[i + j * ldc], ldc, alpha, beta, ithr);
            } else {
                kernel_mxn<isTransA, isTransB>(K, a, lda, b, ldb,
                        &C[i + j * ldc], ldc, alpha, beta, ithr);
            }
        }
    }

    // tail processing
    for (dim_t i = 0; i < M; i++) {
        for (dim_t j = Nu; j < N; j++) {
            float c = beta == 0.f ? 0.f : beta * C[i + j * ldc];
            for (dim_t p = 0; p < K; p++) {
                float b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                float a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
    for (dim_t i = Mu; i < M; i++) {
        for (dim_t j = 0; j < Nu; j++) {
            float c = beta == 0.f ? 0.f : beta * C[i + j * ldc];
            for (dim_t p = 0; p < K; p++) {
                float b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                float a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
}

template <bool isTransA, bool isTransB>
void gemm_ithr(const dim_t M, const dim_t N, const dim_t K, const float alpha,
        const float *A, const dim_t lda, const float *B, const dim_t ldb,
        const float beta, float *C, const dim_t ldc, bool do_copy, float *ws,
        int ithr = -1) {

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
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, beta, ws, do_copy, ithr);
                } else {
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, 1.0f, ws, do_copy, ithr);
                }
            }
        }
    }
}
} // namespace

dnnl_status_t rvv_gemm_f32(const char *transa_, const char *transb_,
        const dim_t *M_, const dim_t *N_, const dim_t *K_, const float *alpha_,
        const float *A, const dim_t *lda_, const float *B, const dim_t *ldb_,
        const float *beta_, float *C, const dim_t *ldc_, const float *bias) {

    if (!(utils::one_of(*transa_, 'n', 'N', 't', 'T')
                && utils::one_of(*transb_, 'n', 'N', 't', 'T')))
        return dnnl_unimplemented;

    bool isTransA = (*transa_ == 'T' || *transa_ == 't');
    bool isTransB = (*transb_ == 'T' || *transb_ == 't');

    if (isTransA && !isTransB) {
        return ref_gemm<float>(transa_, transb_, M_, N_, K_, alpha_, A, lda_, B,
                ldb_, beta_, C, ldc_, bias);
    }

    const dim_t M = *M_, N = *N_, K = *K_;
    const dim_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const float alpha = *alpha_, beta = *beta_;

    // early out and avoid division by zero in partitioning
    if (utils::one_of(0, M, N)) return dnnl_success;

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

    bool do_copy = (NB / unroll_factor<float>::n > 3);
    const int nthr_mn = nthr_m * nthr_n;
    const int nthr_to_use = nthr_mn * nthr_k;
    const size_t ws_elems_per_thr = K * unroll_factor<float>::m;
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

            if (!isTransA) {
                if (!isTransB) {
                    gemm_ithr<false, false>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws, ithr);
                } else {
                    gemm_ithr<false, true>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws, ithr);
                }
            } else {
                if (!isTransB) {
                    gemm_ithr<true, false>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws, ithr);
                } else {
                    gemm_ithr<true, true>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws, ithr);
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

    if (bias) {
        parallel_nd(N, M, [&](dim_t i, dim_t j) { C[i * ldc + j] += bias[j]; });
    }

    free(ws_buffers);
    free(c_buffers);

    return dnnl_success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl