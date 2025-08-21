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
#include "common/bfloat16.hpp"

#include "cpu/platform.hpp"

#include "cpu/rv64/gemm/rvv_gemm_bf16.hpp"
#include "cpu/rv64/gemm/rvv_gemm_utils_bf16.hpp"

#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::utils;
using namespace gemm_utils;

namespace {

inline void rvv_cvt_bf16_to_f32_vector(float *out, const bfloat16_t *inp, size_t nelems) {
    size_t i = 0;
    while (i < nelems) {
        size_t vl = __riscv_vsetvl_e16m1(nelems - i);

        vuint16m1_t v_bf16 = __riscv_vle16_v_u16m1((const uint16_t*)(inp + i), vl);

        vuint32m2_t v_f32_bits = __riscv_vzext_vf2_u32m2(v_bf16, vl);
        v_f32_bits = __riscv_vsll_vx_u32m2(v_f32_bits, 16, vl);

        vfloat32m2_t v_f32 = __riscv_vreinterpret_v_u32m2_f32m2(v_f32_bits);

        __riscv_vse32_v_f32m2(out + i, v_f32, vl);

        i += vl;
    }
}

void copy_A(bool isTransA, dim_t K, const bfloat16_t *A, const dim_t lda, float *ws) {
    constexpr dim_t m = unroll_factor_bf16<bfloat16_t>::m;

    for (dim_t k = 0; k < K; k++) {
        if (isTransA) {
            dim_t i = 0;
            while (i < m) {
                size_t vl = __riscv_vsetvl_e16m1(m - i);
                ptrdiff_t stride = lda * sizeof(bfloat16_t);
                const bfloat16_t *a_ptr = A + i * lda + k;

                vuint16m1_t v_a_bf16 = __riscv_vlse16_v_u16m1(
                    (const uint16_t*)a_ptr, stride, vl);

                vuint32m2_t v_a_f32_bits = __riscv_vzext_vf2_u32m2(v_a_bf16, vl);
                v_a_f32_bits = __riscv_vsll_vx_u32m2(v_a_f32_bits, 16, vl);
                vfloat32m2_t v_a_f32 = __riscv_vreinterpret_v_u32m2_f32m2(v_a_f32_bits);

                __riscv_vse32_v_f32m2(ws + i, v_a_f32, vl);
                i += vl;
            }
        } else {
            const bfloat16_t *a_ptr = A + k * lda;
            rvv_cvt_bf16_to_f32_vector(ws, a_ptr, m);
        }
        ws += m;
    }
}

template <bool isTransA, bool isTransB>
void kernel_mxn(dim_t K, const bfloat16_t *A, const dim_t lda, const bfloat16_t *B,
        const dim_t ldb, float *C, const dim_t ldc, const float alpha,
        const float beta, int ithr = -1) {
    constexpr dim_t m = unroll_factor_bf16<bfloat16_t>::m;
    constexpr dim_t n = unroll_factor_bf16<bfloat16_t>::n;

    float c[m * n] = {0.0f};

    for (dim_t k = 0; k < K; k++) {
        dim_t i = 0;
        while (i < m) {
            size_t vl = __riscv_vsetvl_e16m1(m - i);
            vfloat32m2_t v_a;

            if (isTransA) {
                ptrdiff_t stride_a = lda * sizeof(bfloat16_t);
                vuint16m1_t v_a_bf16 = __riscv_vlse16_v_u16m1(
                    (const uint16_t*)(A + i * lda + k), stride_a, vl);

                vuint32m2_t v_a_f32_bits = __riscv_vzext_vf2_u32m2(v_a_bf16, vl);
                v_a_f32_bits = __riscv_vsll_vx_u32m2(v_a_f32_bits, 16, vl);
                v_a = __riscv_vreinterpret_v_u32m2_f32m2(v_a_f32_bits);
            } else {
                vuint16m1_t v_a_bf16 = __riscv_vle16_v_u16m1(
                    (const uint16_t*)(A + i + k * lda), vl);

                vuint32m2_t v_a_f32_bits = __riscv_vzext_vf2_u32m2(v_a_bf16, vl);
                v_a_f32_bits = __riscv_vsll_vx_u32m2(v_a_f32_bits, 16, vl);
                v_a = __riscv_vreinterpret_v_u32m2_f32m2(v_a_f32_bits);
            }

            for (dim_t j = 0; j < n; j++) {
                bfloat16_t b_bf16 = isTransB ? B[j + k * ldb] : B[k + j * ldb];
                float b = static_cast<float>(b_bf16);

                float *c_col_ptr = c + m * j + i;
                vfloat32m2_t v_c = __riscv_vle32_v_f32m2(c_col_ptr, vl);

                v_c = __riscv_vfmacc_vf_f32m2(v_c, b, v_a, vl);
                __riscv_vse32_v_f32m2(c_col_ptr, v_c, vl);
            }
            i += vl;
        }
    }

    for (dim_t j = 0; j < n; j++) {
        dim_t i = 0;
        while (i < m) {
            size_t vl = __riscv_vsetvl_e32m2(m - i);

            float *c_final_ptr = C + j * ldc + i;
            float *c_acc_ptr = c + j * m + i;

            vfloat32m2_t v_acc = __riscv_vle32_v_f32m2(c_acc_ptr, vl);
            vfloat32m2_t v_res;

            if (beta == 0.0f) {
                v_res = __riscv_vfmul_vf_f32m2(v_acc, alpha, vl);
            } else {
                vfloat32m2_t v_c_old = __riscv_vle32_v_f32m2(c_final_ptr, vl);
                v_res = __riscv_vfmul_vf_f32m2(v_c_old, beta, vl);
                v_res = __riscv_vfmacc_vf_f32m2(v_res, alpha, v_acc, vl);
            }

            __riscv_vse32_v_f32m2(c_final_ptr, v_res, vl);
            i += vl;
        }
    }
}

template <bool isTransA, bool isTransB>
void block_ker(const dim_t M, const dim_t N, const dim_t K, const bfloat16_t *A,
        const dim_t lda, const bfloat16_t *B, const dim_t ldb, float *C,
        const dim_t ldc, const float alpha, const float beta, float *ws,
        bool do_copy, int ithr = -1) {

    constexpr dim_t m = unroll_factor_bf16<bfloat16_t>::m;
    constexpr dim_t n = unroll_factor_bf16<bfloat16_t>::n;
    dim_t Nu = (N / n) * n;
    dim_t Mu = (M / m) * m;

    for (dim_t i = 0; i < Mu; i += m) {
        for (dim_t j = 0; j < Nu; j += n) {
            const bfloat16_t *b = isTransB ? &B[j] : &B[j * ldb];
            const bfloat16_t *a = isTransA ? &A[i * lda] : &A[i];
            if (do_copy) {
                if (j == 0) { copy_A(isTransA, K, a, lda, ws); }
                kernel_mxn<isTransA, isTransB>(K, a, lda, b, ldb,
                        &C[i + j * ldc], ldc, alpha, beta, ithr);
            } else {
                kernel_mxn<isTransA, isTransB>(K, a, lda, b, ldb,
                        &C[i + j * ldc], ldc, alpha, beta, ithr);
            }
        }
    }

    for (dim_t i = 0; i < M; i++) {
        for (dim_t j = Nu; j < N; j++) {
            float c = beta == 0.f ? 0.f : beta * C[i + j * ldc];

            for (dim_t p = 0; p < K; p++) {
                float b = static_cast<float>(isTransB ? B[j + p * ldb] : B[p + j * ldb]);
                float a = static_cast<float>(isTransA ? A[p + i * lda] : A[i + p * lda]);
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
    for (dim_t i = Mu; i < M; i++) {
        for (dim_t j = 0; j < Nu; j++) {
            float c = beta == 0.f ? 0.f : beta * C[i + j * ldc];

            for (dim_t p = 0; p < K; p++) {
                float b = static_cast<float>(isTransB ? B[j + p * ldb] : B[p + j * ldb]);
                float a = static_cast<float>(isTransA ? A[p + i * lda] : A[i + p * lda]);
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
}

template <bool isTransA, bool isTransB>
void gemm_ithr(const dim_t M, const dim_t N, const dim_t K, const float alpha,
        const bfloat16_t *A, const dim_t lda, const bfloat16_t *B, const dim_t ldb,
        const float beta, float *C, const dim_t ldc, bool do_copy, float *ws,
        int ithr = -1) {

    constexpr dim_t BM = gemm_traits_t<bfloat16_t, isTransA, isTransB>::BM;
    constexpr dim_t BN = gemm_traits_t<bfloat16_t, isTransA, isTransB>::BN;
    constexpr dim_t BK = gemm_traits_t<bfloat16_t, isTransA, isTransB>::BK;

    const bfloat16_t *curA;
    const bfloat16_t *curB;
    float *curC;

    if ((M <= 0) || (N <= 0)) return;

    if ((K <= 0) || (alpha == 0.f)) {
        dim_t MN = N * M;
        if (beta == 0.f) {
            dim_t j = 0;
            while (j < MN) {
                size_t vl = __riscv_vsetvl_e32m1(MN - j);
                vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                __riscv_vse32_v_f32m1(C + j, v_zero, vl);
                j += vl;
            }
        } else if (beta != 1.f) {
            dim_t j = 0;
            while (j < MN) {
                size_t vl = __riscv_vsetvl_e32m1(MN - j);
                vfloat32m1_t v_c = __riscv_vle32_v_f32m1(C + j, vl);
                v_c = __riscv_vfmul_vf_f32m1(v_c, beta, vl);
                __riscv_vse32_v_f32m1(C + j, v_c, vl);
                j += vl;
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
                curA = isTransA ? &A[Bk + Bm * lda] : &A[Bm + Bk * lda];
                curB = isTransB ? &B[Bn + Bk * ldb] : &B[Bk + Bn * ldb];
                curC = &C[Bm + Bn * ldc];

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

dnnl_status_t rvv_gemm_bf16bf16f32(const char *transa_, const char *transb_,
        const dim_t *M_, const dim_t *N_, const dim_t *K_, const float *alpha_,
        const bfloat16_t *A, const dim_t *lda_, const bfloat16_t *B, const dim_t *ldb_,
        const float *beta_, float *C, const dim_t *ldc_) {

    if (!(utils::one_of(*transa_, 'n', 'N', 't', 'T')
                && utils::one_of(*transb_, 'n', 'N', 't', 'T')))
        return dnnl_unimplemented;

    bool isTransA = (*transa_ == 'T' || *transa_ == 't');
    bool isTransB = (*transb_ == 'T' || *transb_ == 't');
    const dim_t M = *M_, N = *N_, K = *K_;
    const dim_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const float alpha = *alpha_, beta = *beta_;

    if (utils::one_of(0, M, N)) return dnnl_success;

    const bool do_copy = false;
    float *ws = nullptr;

    if (!isTransA && !isTransB) {
        gemm_ithr<false, false>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, do_copy, ws);
    } else if (!isTransA && isTransB) {
        gemm_ithr<false, true>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, do_copy, ws);
    } else if (isTransA && !isTransB) {
        gemm_ithr<true, false>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, do_copy, ws);
    } else {
        gemm_ithr<true, true>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, do_copy, ws);
    }

    return dnnl_success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl