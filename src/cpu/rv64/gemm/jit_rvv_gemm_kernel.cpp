/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#include "cpu/rv64/gemm/jit_rvv_gemm_kernel.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

using namespace Xbyak_riscv;

jit_rvv_gemm_kernel_t::jit_rvv_gemm_kernel_t(
        dim_t n_cols, bool isTransA, bool isTransB, bool has_bias)
    : jit_generator_t("rv64_gemm_kernel_f32_jit")
    , n_cols_(n_cols)
    , isTransA_(isTransA)
    , isTransB_(isTransB)
    , has_bias_(has_bias) {
    create_kernel();
}

void jit_rvv_gemm_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;

    const Reg reg_A_ptr = a1; // running pointer into A
    const Reg reg_m = a2; // tile height (used for vsetvli)
    const Reg reg_C_base = a3; // base pointer to C(:, 0)

    const Reg reg_lda_bytes = t0;
    const Reg reg_ldb_bytes = t1;
    const Reg reg_ldc_bytes = t2;
    const Reg reg_K = t3;
    const Reg reg_alpha_bits = t4;
    const Reg reg_bias_ptr = t4; // reuse after alpha bits moved to freg
    const Reg reg_beta_bits = t5;

    const Reg reg_k = a4; // current k counter
    const Reg reg_K_main = a5; // (K / 4) * 4
    const Reg reg_B0_ptr = a6; // running pointer into B
    const Reg reg_tmp0 = a7;
    const FReg freg_alpha = fa0;
    const FReg freg_beta = fa1;
    const FReg freg_b[7] = {fa2, fa3, fa4, fa5, fa6, fa7, ft0};

    const VReg v_c[7] = {
            VReg(0), VReg(4), VReg(8), VReg(12), VReg(16), VReg(20), VReg(24)};
    const VReg v_a(28);

    // Layout of call_params_t:
    //   0  : const float *A
    //   8  : const float *B
    //   16 : float *C
    //   24 : dim_t lda
    //   32 : dim_t ldb
    //   40 : dim_t ldc
    //   48 : dim_t K
    //   56 : dim_t m
    //   64 : float alpha
    //   68 : float beta
    //   72 : const float *bias  (only used when has_bias_)
    ld(reg_A_ptr, reg_param, 0);
    ld(reg_B0_ptr, reg_param, 8);
    ld(reg_C_base, reg_param, 16);
    ld(reg_lda_bytes, reg_param, 24);
    ld(reg_ldb_bytes, reg_param, 32);
    ld(reg_ldc_bytes, reg_param, 40);
    ld(reg_K, reg_param, 48);
    ld(reg_m, reg_param, 56);

    lw(reg_alpha_bits, reg_param, 64);
    fmv_w_x(freg_alpha, reg_alpha_bits);
    lw(reg_beta_bits, reg_param, 68);
    fmv_w_x(freg_beta, reg_beta_bits);

    if (has_bias_) { ld(reg_bias_ptr, reg_param, 72); }

    slli(reg_lda_bytes, reg_lda_bytes, 2);
    slli(reg_ldb_bytes, reg_ldb_bytes, 2);
    slli(reg_ldc_bytes, reg_ldc_bytes, 2);

    vsetvli(x0, reg_m, SEW::e32, LMUL::m4);

    const Reg &reg_tmp3 = reg_param;

    for (dim_t c = 0; c < n_cols_; c++)
        vmv_v_i(v_c[c], 0);

    mv(reg_K_main, reg_K);
    srli(reg_tmp3, reg_K_main, 2);
    slli(reg_K_main, reg_tmp3, 2);

    auto emit_k_step = [&]() {
        if (isTransA_) {
            vlse32_v(v_a, reg_A_ptr, reg_lda_bytes);
        } else {
            vle32_v(v_a, reg_A_ptr);
        }

        if (isTransB_) {
            for (dim_t c = 0; c < n_cols_; c++) {
                flw(freg_b[c], reg_B0_ptr, static_cast<int32_t>(c * 4));
            }
        } else {
            flw(freg_b[0], reg_B0_ptr, 0);
            if (n_cols_ > 1) {
                add(reg_tmp0, reg_B0_ptr, reg_ldb_bytes);
                flw(freg_b[1], reg_tmp0, 0);
                for (dim_t c = 2; c < n_cols_; c++) {
                    add(reg_tmp0, reg_tmp0, reg_ldb_bytes);
                    flw(freg_b[c], reg_tmp0, 0);
                }
            }
        }

        for (dim_t c = 0; c < n_cols_; c++)
            vfmacc_vf(v_c[c], freg_b[c], v_a);

        if (isTransA_) {
            addi(reg_A_ptr, reg_A_ptr, 4);
        } else {
            add(reg_A_ptr, reg_A_ptr, reg_lda_bytes);
        }

        if (isTransB_) {
            add(reg_B0_ptr, reg_B0_ptr, reg_ldb_bytes);
        } else {
            addi(reg_B0_ptr, reg_B0_ptr, 4);
        }
    };

    mv(reg_k, x0);

    Label label_k_main_loop, label_k_main_end;
    Label label_k_tail_loop, label_k_tail_end;

    L(label_k_main_loop);
    bge(reg_k, reg_K_main, label_k_main_end);

    emit_k_step();
    emit_k_step();
    emit_k_step();
    emit_k_step();

    addi(reg_k, reg_k, 4);
    j_(label_k_main_loop);

    L(label_k_main_end);

    // Tail K loop for K % 4
    L(label_k_tail_loop);
    bge(reg_k, reg_K, label_k_tail_end);

    emit_k_step();

    addi(reg_k, reg_k, 1);
    j_(label_k_tail_loop);

    L(label_k_tail_end);

    if (has_bias_) {
        // C-update with fused bias: result = alpha*acc + beta*C + bias
        auto emit_c_update = [&](dim_t col_idx) {
            Label label_beta_zero, label_done;
            Label label_skip_bias, label_c_store;

            if (col_idx == 0) {
                mv(reg_tmp3, reg_C_base);
            } else {
                li(reg_tmp0, col_idx);
                mul(reg_tmp3, reg_ldc_bytes, reg_tmp0);
                add(reg_tmp3, reg_C_base, reg_tmp3);
            }

            beq(reg_beta_bits, x0, label_beta_zero);

            vle32_v(v_a, reg_tmp3);
            vfmul_vf(v_a, v_a, freg_beta);
            vfmul_vf(v_c[col_idx], v_c[col_idx], freg_alpha);
            vfadd_vv(v_a, v_a, v_c[col_idx]);

            beq(reg_bias_ptr, x0, label_skip_bias);
            vle32_v(v_c[col_idx], reg_bias_ptr);
            vfadd_vv(v_a, v_a, v_c[col_idx]);
            L(label_skip_bias);

            vse32_v(v_a, reg_tmp3);
            j_(label_done);

            L(label_beta_zero);
            vfmul_vf(v_c[col_idx], v_c[col_idx], freg_alpha);

            beq(reg_bias_ptr, x0, label_c_store);
            vle32_v(v_a, reg_bias_ptr);
            vfadd_vv(v_c[col_idx], v_c[col_idx], v_a);

            L(label_c_store);
            vse32_v(v_c[col_idx], reg_tmp3);

            L(label_done);
        };

        for (dim_t c = 0; c < n_cols_; c++)
            emit_c_update(c);
    } else {
        // C-update without bias: result = alpha*acc + beta*C
        auto emit_c_update = [&](dim_t col_idx) {
            Label label_beta_zero, label_done;

            if (col_idx == 0) {
                mv(reg_tmp3, reg_C_base);
            } else {
                li(reg_tmp0, col_idx);
                mul(reg_tmp3, reg_ldc_bytes, reg_tmp0);
                add(reg_tmp3, reg_C_base, reg_tmp3);
            }

            beq(reg_beta_bits, x0, label_beta_zero);

            vle32_v(v_a, reg_tmp3);
            vfmul_vf(v_a, v_a, freg_beta);
            vfmul_vf(v_c[col_idx], v_c[col_idx], freg_alpha);
            vfadd_vv(v_a, v_a, v_c[col_idx]);
            vse32_v(v_a, reg_tmp3);
            j_(label_done);

            L(label_beta_zero);
            vfmul_vf(v_c[col_idx], v_c[col_idx], freg_alpha);
            vse32_v(v_c[col_idx], reg_tmp3);

            L(label_done);
        };

        for (dim_t c = 0; c < n_cols_; c++)
            emit_c_update(c);
    }

    ret();
#else
    ret();
#endif
}

namespace {

template <bool isTransA, bool isTransB>
void jit_rvv_gemm_kernel_dispatch(const float *A, const float *B, float *C,
        dim_t lda, dim_t ldb, dim_t ldc, dim_t K, float alpha, float beta,
        dim_t m, dim_t n_cols, const float *bias) {
    // Kernels without fused bias (same code size as upstream)
    static jit_rvv_gemm_kernel_t nb1(1, isTransA, isTransB, false);
    static jit_rvv_gemm_kernel_t nb2(2, isTransA, isTransB, false);
    static jit_rvv_gemm_kernel_t nb3(3, isTransA, isTransB, false);
    static jit_rvv_gemm_kernel_t nb4(4, isTransA, isTransB, false);
    static jit_rvv_gemm_kernel_t nb5(5, isTransA, isTransB, false);
    static jit_rvv_gemm_kernel_t nb6(6, isTransA, isTransB, false);
    static jit_rvv_gemm_kernel_t nb7(7, isTransA, isTransB, false);

    static jit_rvv_gemm_kernel_t *arr_nb[]
            = {nullptr, &nb1, &nb2, &nb3, &nb4, &nb5, &nb6, &nb7};

    // Kernels with fused bias
    static jit_rvv_gemm_kernel_t b1(1, isTransA, isTransB, true);
    static jit_rvv_gemm_kernel_t b2(2, isTransA, isTransB, true);
    static jit_rvv_gemm_kernel_t b3(3, isTransA, isTransB, true);
    static jit_rvv_gemm_kernel_t b4(4, isTransA, isTransB, true);
    static jit_rvv_gemm_kernel_t b5(5, isTransA, isTransB, true);
    static jit_rvv_gemm_kernel_t b6(6, isTransA, isTransB, true);
    static jit_rvv_gemm_kernel_t b7(7, isTransA, isTransB, true);

    static jit_rvv_gemm_kernel_t *arr_b[]
            = {nullptr, &b1, &b2, &b3, &b4, &b5, &b6, &b7};

    static bool verbose_printed = false;
    if (!verbose_printed) {
        VINFO(primitive, create, dispatch, rvv_gemm_jit,
                "JIT gemm kernel taking over: m=%d, n=%d", (int)m, (int)n_cols);
        verbose_printed = true;
    }

    jit_rvv_gemm_kernel_t::call_params_t p;
    p.A = A;
    p.B = B;
    p.C = C;
    p.lda = lda;
    p.ldb = ldb;
    p.ldc = ldc;
    p.K = K;
    p.m = m;
    p.alpha = alpha;
    p.beta = beta;
    p.bias = bias;

    jit_rvv_gemm_kernel_t **arr = bias ? arr_b : arr_nb;
    (*arr[n_cols])(&p);
}

} // namespace

void jit_rvv_gemm_kernel(const float *A, const float *B, float *C, dim_t lda,
        dim_t ldb, dim_t ldc, dim_t K, float alpha, float beta, dim_t m,
        dim_t n_cols, bool isTransA, bool isTransB, const float *bias) {
    if (!isTransA && !isTransB) {
        jit_rvv_gemm_kernel_dispatch<false, false>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n_cols, bias);
    } else if (isTransA && !isTransB) {
        jit_rvv_gemm_kernel_dispatch<true, false>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n_cols, bias);
    } else if (!isTransA && isTransB) {
        jit_rvv_gemm_kernel_dispatch<false, true>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n_cols, bias);
    } else {
        jit_rvv_gemm_kernel_dispatch<true, true>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n_cols, bias);
    }
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
