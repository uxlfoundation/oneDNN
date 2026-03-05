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

jit_rvv_gemm_kernel_t::jit_rvv_gemm_kernel_t(dim_t m)
    : jit_generator_t("rv64_gemm_kernel_f32_jit"), m_(m) {
    create_kernel();
}

void jit_rvv_gemm_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;

    // Persistent scalar parameters and pointers:
    const Reg reg_A_ptr = a1; // running pointer over rows of A
    const Reg reg_C_base = a3; // base pointer to C(:, 0)

    const Reg reg_lda_bytes = t0; // lda * sizeof(float)
    const Reg reg_ldb_bytes = t1; // ldb * sizeof(float)
    const Reg reg_ldc_bytes = t2; // ldc * sizeof(float)
    const Reg reg_K = t3; // K
    const Reg reg_alpha_bits
            = t4; // holds raw alpha bits, loaded and then moved to freg_alpha for computation
    const Reg reg_beta_bits = t5; // raw beta bits (0 means beta == 0.0f)

    // Loop counters:
    const Reg reg_k = a4; // current k
    const Reg reg_K_main = a5; // K_main = (K / 4) * 4
    // B row pointer for current k (points to B(k, 0)).
    const Reg reg_B0_ptr = a6; // &B(k, 0)

    // Scratch temporaries used for address arithmetic.
    const Reg reg_tmp0 = a0; // reused after initial parameter loads
    const Reg reg_tmp1 = t6;

    const FReg freg_alpha = fa0;
    const FReg freg_beta = fa1;
    const FReg freg_b0 = fa2;
    const FReg freg_b1 = fa3;
    const FReg freg_b2 = fa4;
    const FReg freg_b3 = fa5;
    const FReg freg_b4 = fa6;
    const FReg freg_b5 = fa7;
    const FReg freg_b6 = ft0;

// Vector register layout (LMUL=m4, f32):
//   7 accumulators + 1 A vector == 8 groups -> 8 * 4 = 32 vector registers.
//   v_c0(0), v_c1(4), v_c2(8), v_c3(12), v_c4(16), v_c5(20), v_c6(24), v_a0(28)
const VReg v_c0(0);
const VReg v_c1(4);
const VReg v_c2(8);
const VReg v_c3(12);
const VReg v_c4(16);
const VReg v_c5(20);
const VReg v_c6(24);
const VReg v_a0(28);

// Layout of call_params_t:

    //   0  : const float *A;
    //   8  : const float *B;
    //   16 : float *C;
    //   24 : dim_t lda;
    //   32 : dim_t ldb;
    //   40 : dim_t ldc;
    //   48 : dim_t K;
    //   56 : float alpha;
    //   60 : float beta;
    ld(reg_A_ptr, reg_param, 0);
    ld(reg_B0_ptr, reg_param, 8); // B base
    ld(reg_C_base, reg_param, 16);
    ld(reg_lda_bytes, reg_param, 24);
    ld(reg_ldb_bytes, reg_param, 32);
    ld(reg_ldc_bytes, reg_param, 40);
    ld(reg_K, reg_param, 48);

    // Convert leading dimensions from elements to bytes.
    slli(reg_lda_bytes, reg_lda_bytes, 2);
    slli(reg_ldb_bytes, reg_ldb_bytes, 2);
    slli(reg_ldc_bytes, reg_ldc_bytes, 2);

    // Load alpha / beta as raw bits and as FP registers.
    lw(reg_alpha_bits, reg_param, 56);
    fmv_w_x(freg_alpha, reg_alpha_bits);

    lw(reg_beta_bits, reg_param, 60);
    fmv_w_x(freg_beta, reg_beta_bits);

    // Set VL once for m_ rows; with LMUL = m2 (for m=8) or m4 (for m=16)
    const LMUL lmul = LMUL::m4;
    li(reg_tmp0, m_);
    vsetvli(x0, reg_tmp0, SEW::e32, lmul);
    // B0_ptr already points to B(k=0, 0).

    // Zero accumulators for 7 output columns.
    vmv_v_i(v_c0, 0);
    vmv_v_i(v_c1, 0);
    vmv_v_i(v_c2, 0);
    vmv_v_i(v_c3, 0);
    vmv_v_i(v_c4, 0);
    vmv_v_i(v_c5, 0);
    vmv_v_i(v_c6, 0);

    // Base pointers for this 8-row micro-tile.
    // A_ptr starts at A(:, 0) and advances by lda_bytes each FMA step.
    // B0_ptr/B1_ptr start at B(k=0, 0/1) and advance by 4 bytes (next k).

    // K_main = (K / 4) * 4
    mv(reg_K_main, reg_K);
    srli(reg_tmp0, reg_K_main, 2);
    slli(reg_K_main, reg_tmp0, 2);

    // Helper lambda to emit one K-step of the unrolled micro-kernel.
    // It keeps the exact instruction sequence as the inlined version:
    //   - load A row vector
    //   - load 4 B scalars (one per output column)
    //   - 4 FMAs into v_c0..v_c3
    //   - advance A_ptr and B0/B1 pointers
    
auto emit_k_step = [&]() {
    vle32_v(v_a0, reg_A_ptr);

    // Load 7 scalars from B(k, 0..6) with stride ldb_bytes.
    mv(reg_tmp1, reg_B0_ptr);
    flw(freg_b0, reg_tmp1, 0);
    add(reg_tmp1, reg_tmp1, reg_ldb_bytes);
    flw(freg_b1, reg_tmp1, 0);
    add(reg_tmp1, reg_tmp1, reg_ldb_bytes);
    flw(freg_b2, reg_tmp1, 0);
    add(reg_tmp1, reg_tmp1, reg_ldb_bytes);
    flw(freg_b3, reg_tmp1, 0);
    add(reg_tmp1, reg_tmp1, reg_ldb_bytes);
    flw(freg_b4, reg_tmp1, 0);
    add(reg_tmp1, reg_tmp1, reg_ldb_bytes);
    flw(freg_b5, reg_tmp1, 0);
    add(reg_tmp1, reg_tmp1, reg_ldb_bytes);
    flw(freg_b6, reg_tmp1, 0);

    vfmacc_vf(v_c0, freg_b0, v_a0);
    vfmacc_vf(v_c1, freg_b1, v_a0);
    vfmacc_vf(v_c2, freg_b2, v_a0);
    vfmacc_vf(v_c3, freg_b3, v_a0);
    vfmacc_vf(v_c4, freg_b4, v_a0);
    vfmacc_vf(v_c5, freg_b5, v_a0);
    vfmacc_vf(v_c6, freg_b6, v_a0);

    add(reg_A_ptr, reg_A_ptr, reg_lda_bytes);
    addi(reg_B0_ptr, reg_B0_ptr, 4);
};


    // k = 0
    mv(reg_k, x0);

    Label label_k_main_loop, label_k_main_end;
    Label label_k_tail_loop, label_k_tail_end;

    // Main K loop: 4-way unrolled with pointer strength reduction.
    L(label_k_main_loop);
    bge(reg_k, reg_K_main, label_k_main_end);

    // ---- k, k+1, k+2, k+3 ----
    emit_k_step();
    emit_k_step();
    emit_k_step();
    emit_k_step();

    addi(reg_k, reg_k, 4);
    j_(label_k_main_loop);

    L(label_k_main_end);

    // Tail K loop for K % 4.
    L(label_k_tail_loop);
    bge(reg_k, reg_K, label_k_tail_end);

    emit_k_step();

    addi(reg_k, reg_k, 1);
    j_(label_k_tail_loop);

    L(label_k_tail_end);

    // Combine accumulators with C: C = alpha * accum + beta * C.
    
auto emit_c_update = [&](int col_idx, const VReg &v_c) {
    Label label_beta_zero, label_done;
    Reg reg_c_col = reg_tmp0;

    if (col_idx == 0) {
        mv(reg_c_col, reg_C_base);
    } else {
        li(reg_tmp1, col_idx);
        mul(reg_c_col, reg_ldc_bytes, reg_tmp1);
        add(reg_c_col, reg_C_base, reg_c_col);
    }

    // beta == 0.0f
    beq(reg_beta_bits, x0, label_beta_zero);

    // beta != 0: tmp(v_a0) = beta * C_old + alpha * v_c
    vle32_v(v_a0, reg_c_col);
    vfmul_vf(v_a0, v_a0, freg_beta);
    vfmul_vf(v_c, v_c, freg_alpha);
    vfadd_vv(v_a0, v_a0, v_c);
    vse32_v(v_a0, reg_c_col);
    j_(label_done);

    // beta == 0: v_c = alpha * v_c
    L(label_beta_zero);
    vfmul_vf(v_c, v_c, freg_alpha);
    vse32_v(v_c, reg_c_col);

    L(label_done);
};

emit_c_update(0, v_c0);
emit_c_update(1, v_c1);
emit_c_update(2, v_c2);
emit_c_update(3, v_c3);
emit_c_update(4, v_c4);
emit_c_update(5, v_c5);
emit_c_update(6, v_c6);
    ret();
#else
    // RVV JIT is disabled, emit a stub. This kernel should never be used
    // when RVV is not available at build time.
    ret();
#endif
}

void jit_rvv_gemm_kernel(const float *A, const float *B, float *C, dim_t lda,
        dim_t ldb, dim_t ldc, dim_t K, float alpha, float beta, dim_t m) {
    static jit_rvv_gemm_kernel_t kernel_m16(16);
    static jit_rvv_gemm_kernel_t kernel_m32(32);
    static jit_rvv_gemm_kernel_t kernel_m64(64);
    static jit_rvv_gemm_kernel_t kernel_m128(128);

    // Print verbose message to indicate JIT kernel is being used for GEMM
    static bool verbose_printed = false;
    if (!verbose_printed) {
        VINFO(primitive, create, dispatch, rvv_gemm_jit,
                "JIT gemm kernel taking over: m=%d, n=7", (int)m);
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
    p.alpha = alpha;
    p.beta = beta;

    switch ((int)m) {
        case 16: kernel_m16(&p); break;
        case 32: kernel_m32(&p); break;
        case 64: kernel_m64(&p); break;
        case 128: kernel_m128(&p); break;
        default: kernel_m16(&p); break;
    }
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
