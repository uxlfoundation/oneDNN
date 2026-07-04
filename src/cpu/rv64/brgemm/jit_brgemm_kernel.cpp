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

#include "cpu/rv64/brgemm/jit_brgemm_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

struct jit_brgemm_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_t)

    jit_brgemm_kernel_t(const brgemm_desc_t &brg)
        : jit_generator_t("rv64_brgemm_kernel_f32_jit"), brg_(brg) {}

    void operator()(brgemm_kernel_params_t *p) const {
        jit_generator_t::operator()(p);
    }

    const brgemm_desc_t &get_brg() const { return brg_; }

protected:
    void generate() override;

private:
    brgemm_desc_t brg_;
};

void jit_brgemm_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const dim_t LDA_bytes = brg_.LDA * brg_.typesize_A;
    const dim_t LDB_bytes = brg_.LDB * brg_.typesize_B;
    const dim_t LDC_bytes = brg_.LDC * brg_.typesize_C;
    const dim_t N_stride_B = 4 * LDB_bytes; // B advance per 4-col group
    const dim_t N_stride_C = 4 * LDC_bytes; // C advance per 4-col group

    // If all B column offsets (0, LDB, 2*LDB, 3*LDB in bytes) fit in the
    // 12-bit signed immediate of flw, use a single B pointer with immediate
    // offsets instead of 4 separate column pointers.  This saves 3 addi per
    // K step and 3 add at N-loop setup.
    const bool use_single_b = (3 * LDB_bytes <= 2047);

    const Reg reg_param = a0;
    const Reg reg_tmp0 = a0; // reused after param loads

    const Reg reg_A = a1; // A running pointer (advances per K step)
    const Reg reg_n = a2; // N loop counter
    const Reg reg_C = a3; // C column pointer (advances per N group)
    const Reg reg_k = a4; // K loop counter
    const Reg reg_K_main = a5; // (K/4)*4, computed at runtime
    const Reg reg_B0 = a6; // B column-0 running pointer
    const Reg reg_B1 = a7; // B column-1 running pointer

    const Reg reg_lda = t0; // LDA in bytes (baked immediate)
    const Reg reg_ldb = t1; // LDB in bytes (baked immediate)
    const Reg reg_ldc = t2; // LDC in bytes (baked immediate)
    const Reg reg_K_val = t3; // K (runtime, from params)
    const Reg reg_N = t4; // N (from params)
    const Reg reg_beta = t5; // beta raw bits (from params)
    const Reg reg_tmp1 = t6; // scratch

    const Reg reg_A_base = s0; // A base pointer (survives N iterations)
    const Reg reg_B_base = s1; // B group base pointer
    const Reg reg_B2 = s2; // B column-2 running pointer
    const Reg reg_B3 = s3; // B column-3 running pointer
    const Reg reg_bias = s4; // bias vector pointer (callee-saved)

    const FReg freg_b0 = fa0;
    const FReg freg_b1 = fa1;
    const FReg freg_b2 = fa2;
    const FReg freg_b3 = fa3;

    // Vector registers (LMUL=m4: each logical register = 4 physical).
    // 4 accumulators + 2 A-buffers + 1 scratch = 7 groups (28 of 32 phys)
    const VReg v_c0(0); // accum col 0
    const VReg v_c1(4); // accum col 1
    const VReg v_c2(8); // accum col 2
    const VReg v_c3(12); // accum col 3
    const VReg v_a0(16); // A double-buffer 0
    const VReg v_a1(20); // A double-buffer 1
    const VReg v_tmp(24); // scratch for C load/update

    addi(sp, sp, -48);
    sd(reg_A_base, sp, 0);
    sd(reg_B_base, sp, 8);
    sd(reg_B2, sp, 16);
    sd(reg_B3, sp, 24);
    sd(reg_bias, sp, 32);

    ld(reg_A_base, reg_param, 0); // A base
    ld(reg_B_base, reg_param, 8); // B base
    ld(reg_C, reg_param, 16); // C base
    ld(reg_N, reg_param, 24); // N
    ld(reg_tmp1, reg_param, 32); // M (for vsetvli)
    ld(reg_K_val, reg_param, 40); // K (runtime)
    lw(reg_beta, reg_param, 48); // beta bits
    ld(reg_bias, reg_param, 56); // bias pointer

    // Active lanes are overwritten and inactive lanes are never consumed.
    vsetvli(x0, reg_tmp1, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    li(reg_lda, LDA_bytes);
    li(reg_ldb, LDB_bytes);
    li(reg_ldc, LDC_bytes);

    if (use_single_b) {
        // Pre-compute N-stride into callee-saved registers (s2, s3).
        // These replace reg_B2/reg_B3 which are unused in single-B mode.
        li(s2, N_stride_B);
        li(s3, N_stride_C);
    }

    // ---- Compute K_main = (K / 4) * 4 for 4× unrolled loop ----
    srli(reg_K_main, reg_K_val, 2);
    slli(reg_K_main, reg_K_main, 2);

    // ================================================================
    // N outer loop: process 4 columns per iteration
    // ================================================================
    mv(reg_n, x0); // n = 0

    Label lbl_n_loop, lbl_n_tail, lbl_n_tail_loop, lbl_n_end;

    L(lbl_n_loop);
    // Check: n + 4 <= N ?
    addi(reg_tmp0, reg_n, 4);
    blt(reg_N, reg_tmp0, lbl_n_tail); // if N < n+4, go to tail

    mv(reg_A, reg_A_base);

    // ---- Setup B pointer(s) for current N group ----
    mv(reg_B0, reg_B_base);
    if (!use_single_b) {
        add(reg_B1, reg_B_base, reg_ldb);
        add(reg_B2, reg_B1, reg_ldb); // B_base + 2*LDB
        add(reg_B3, reg_B2, reg_ldb); // B_base + 3*LDB
    }

    vmv_v_i(v_c0, 0);
    vmv_v_i(v_c1, 0);
    vmv_v_i(v_c2, 0);
    vmv_v_i(v_c3, 0);

    // ---- K main loop (4× unrolled, double-buffered A loads) ----
    mv(reg_k, x0);
    Label lbl_k_main_end, lbl_k_tail, lbl_k_tail_end;

    // Helper: load 4 B scalars.
    auto emit_b_load = [&]() {
        flw(freg_b0, reg_B0, 0);
        if (use_single_b) {
            flw(freg_b1, reg_B0, LDB_bytes);
            flw(freg_b2, reg_B0, 2 * LDB_bytes);
            flw(freg_b3, reg_B0, 3 * LDB_bytes);
        } else {
            flw(freg_b1, reg_B1, 0);
            flw(freg_b2, reg_B2, 0);
            flw(freg_b3, reg_B3, 0);
        }
    };

    // Helper: advance B pointer(s) to the next k row.
    auto emit_b_advance = [&]() {
        addi(reg_B0, reg_B0, 4);
        if (!use_single_b) {
            addi(reg_B1, reg_B1, 4);
            addi(reg_B2, reg_B2, 4);
            addi(reg_B3, reg_B3, 4);
        }
    };

    // Pipelined K step: prefetch next A into v_next while computing
    // FMAs with the already-loaded v_cur.
    auto emit_pipelined_step = [&](const VReg &v_next, const VReg &v_cur) {
        vle32_v(v_next, reg_A); // prefetch next A
        add(reg_A, reg_A, reg_lda);
        emit_b_load(); // 4 independent scalar loads
        vfmacc_vf(v_c0, freg_b0, v_cur);
        vfmacc_vf(v_c1, freg_b1, v_cur);
        vfmacc_vf(v_c2, freg_b2, v_cur);
        vfmacc_vf(v_c3, freg_b3, v_cur);
        emit_b_advance();
    };

    // Drain step: FMA with v_cur, no prefetch (avoids OOB A read).
    auto emit_drain_step = [&](const VReg &v_cur) {
        emit_b_load();
        vfmacc_vf(v_c0, freg_b0, v_cur);
        vfmacc_vf(v_c1, freg_b1, v_cur);
        vfmacc_vf(v_c2, freg_b2, v_cur);
        vfmacc_vf(v_c3, freg_b3, v_cur);
        emit_b_advance();
    };

    // Guard: skip pipelined section if K_main == 0 (K < 4).
    beq(reg_K_main, x0, lbl_k_tail);

    // Compute K_pipe = K_main - 4 (loop bound for fully-pipelined phase).
    addi(reg_tmp1, reg_K_main, -4); // reg_tmp1 = K_pipe

    // Initial preload of A[0] into v_a0.
    vle32_v(v_a0, reg_A);
    add(reg_A, reg_A, reg_lda);

    align(16);
    {
        Label lbl_pipe, lbl_pipe_end;
        L(lbl_pipe);
        bge(reg_k, reg_tmp1, lbl_pipe_end); // k >= K_pipe → exit

        emit_pipelined_step(v_a1, v_a0); // step 0
        emit_pipelined_step(v_a0, v_a1); // step 1
        emit_pipelined_step(v_a1, v_a0); // step 2
        emit_pipelined_step(v_a0, v_a1); // step 3 (prefetch for next iter)

        addi(reg_k, reg_k, 4);
        j_(lbl_pipe);
        L(lbl_pipe_end);
    }

    // v_a0 holds A[K_pipe] (from initial preload or last step-3).
    emit_pipelined_step(v_a1, v_a0); // rem step 0
    emit_pipelined_step(v_a0, v_a1); // rem step 1
    emit_pipelined_step(v_a1, v_a0); // rem step 2 (last access: A[K_main-1])
    emit_drain_step(v_a1); // rem step 3 (no prefetch → no OOB)
    addi(reg_k, reg_k, 4); // k = K_main

    L(lbl_k_main_end);

    // ---- K tail (1 step at a time, no pipelining needed) ----
    L(lbl_k_tail);
    bge(reg_k, reg_K_val, lbl_k_tail_end);

    vle32_v(v_a0, reg_A);
    add(reg_A, reg_A, reg_lda);
    emit_b_load();
    vfmacc_vf(v_c0, freg_b0, v_a0);
    vfmacc_vf(v_c1, freg_b1, v_a0);
    vfmacc_vf(v_c2, freg_b2, v_a0);
    vfmacc_vf(v_c3, freg_b3, v_a0);
    emit_b_advance();
    addi(reg_k, reg_k, 1);
    j_(lbl_k_tail);

    L(lbl_k_tail_end);

    // ---- Add bias to all accumulators (before C-store) ----
    {
        Label lbl_no_bias;
        beq(reg_bias, x0, lbl_no_bias);
        vle32_v(v_tmp, reg_bias);
        vfadd_vv(v_c0, v_c0, v_tmp);
        vfadd_vv(v_c1, v_c1, v_tmp);
        vfadd_vv(v_c2, v_c2, v_tmp);
        vfadd_vv(v_c3, v_c3, v_tmp);
        L(lbl_no_bias);
    }

    // ---- Store accumulators to C (single beta branch for all cols) ----
    {
        mv(reg_tmp0, reg_C);
        Label lbl_bz, lbl_store_done;
        beq(reg_beta, x0, lbl_bz);

        // beta != 0 path: C[col] = C[col] + accum
        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c0);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c1);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c2);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c3);
        vse32_v(v_tmp, reg_tmp0);

        j_(lbl_store_done);

        // beta == 0 path: C[col] = accum (overwrite)
        L(lbl_bz);
        vse32_v(v_c0, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c1, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c2, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c3, reg_tmp0);

        L(lbl_store_done);
    }

    // ---- Advance B_base and C for next 4-column group ----
    if (use_single_b) {
        add(reg_B_base, reg_B_base, s2); // N_stride_B (precomputed)
        add(reg_C, reg_C, s3); // N_stride_C (precomputed)
    } else {
        li(reg_tmp1, N_stride_B);
        add(reg_B_base, reg_B_base, reg_tmp1);
        li(reg_tmp1, N_stride_C);
        add(reg_C, reg_C, reg_tmp1);
    }

    addi(reg_n, reg_n, 4);
    j_(lbl_n_loop);

    // ================================================================
    // N tail: process 1 column at a time
    // ================================================================
    L(lbl_n_tail);

    L(lbl_n_tail_loop);
    bge(reg_n, reg_N, lbl_n_end);

    // Reset A, setup B0 for single column.
    mv(reg_A, reg_A_base);
    mv(reg_B0, reg_B_base);

    vmv_v_i(v_c0, 0);

    mv(reg_k, x0);
    Label lbl_kt2, lbl_kt2_end;

    L(lbl_kt2);
    bge(reg_k, reg_K_val, lbl_kt2_end);

    vle32_v(v_a0, reg_A);
    flw(freg_b0, reg_B0, 0);
    vfmacc_vf(v_c0, freg_b0, v_a0);
    add(reg_A, reg_A, reg_lda);
    addi(reg_B0, reg_B0, 4);
    addi(reg_k, reg_k, 1);
    j_(lbl_kt2);

    L(lbl_kt2_end);

    // Add bias to accumulator (before single-column C-store).
    {
        Label lbl_no_bias2;
        beq(reg_bias, x0, lbl_no_bias2);
        vle32_v(v_tmp, reg_bias);
        vfadd_vv(v_c0, v_c0, v_tmp);
        L(lbl_no_bias2);
    }

    // Store single column.
    {
        Label lbl_bz2, lbl_done2;
        beq(reg_beta, x0, lbl_bz2);
        vle32_v(v_tmp, reg_C);
        vfadd_vv(v_tmp, v_tmp, v_c0);
        vse32_v(v_tmp, reg_C);
        j_(lbl_done2);
        L(lbl_bz2);
        vse32_v(v_c0, reg_C);
        L(lbl_done2);
    }

    // Advance B_base by 1 column, C by 1 column.
    add(reg_B_base, reg_B_base, reg_ldb);
    add(reg_C, reg_C, reg_ldc);

    addi(reg_n, reg_n, 1);
    j_(lbl_n_tail_loop);

    L(lbl_n_end);

    ld(reg_A_base, sp, 0);
    ld(reg_B_base, sp, 8);
    ld(reg_B2, sp, 16);
    ld(reg_B3, sp, 24);
    ld(reg_bias, sp, 32);
    addi(sp, sp, 48);
    ret();
#else
    // RVV JIT is disabled at build time.
    ret();
#endif
}

brgemm_kernel_common_t::brgemm_kernel_common_t(const brgemm_desc_t &brg)
    : brg_(brg), jit_kernel_(new jit_brgemm_kernel_t(brg)) {}

brgemm_kernel_common_t::~brgemm_kernel_common_t() {
    delete jit_kernel_;
}

status_t brgemm_kernel_common_t::create_kernel() {
    return jit_kernel_->create_kernel();
}

void brgemm_kernel_common_t::operator()(brgemm_kernel_params_t *p) const {
    (*jit_kernel_)(p);
}

// =====================================================================
// f16 JIT kernel: f16 x f16 -> f32 via widening FMA (Zvfh).
// =====================================================================

struct jit_brgemm_f16_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_f16_kernel_t)

    jit_brgemm_f16_kernel_t(const brgemm_desc_t &brg)
        : jit_generator_t("rv64_brgemm_kernel_f16_jit"), brg_(brg) {}

    void operator()(brgemm_kernel_params_t *p) const {
        jit_generator_t::operator()(p);
    }

    const brgemm_desc_t &get_brg() const { return brg_; }

protected:
    void generate() override;

private:
    brgemm_desc_t brg_;
};

void jit_brgemm_f16_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const dim_t LDA_bytes = brg_.LDA * brg_.typesize_A; // f16 → ×2
    const dim_t LDB_bytes = brg_.LDB * brg_.typesize_B; // f16 → ×2
    const dim_t LDC_bytes = brg_.LDC * brg_.typesize_C; // f32 → ×4
    const dim_t N_stride_B = 4 * LDB_bytes;
    const dim_t N_stride_C = 4 * LDC_bytes;

    const bool use_single_b = (3 * LDB_bytes <= 2047);

    const Reg reg_param = a0;
    const Reg reg_tmp0 = a0;

    const Reg reg_A = a1;
    const Reg reg_n = a2;
    const Reg reg_C = a3;
    const Reg reg_k = a4;
    const Reg reg_K_main = a5;
    const Reg reg_B0 = a6;
    const Reg reg_B1 = a7;

    const Reg reg_lda = t0;
    const Reg reg_ldb = t1;
    const Reg reg_ldc = t2;
    const Reg reg_K_val = t3;
    const Reg reg_N = t4;
    const Reg reg_beta = t5;
    const Reg reg_tmp1 = t6;

    const Reg reg_A_base = s0;
    const Reg reg_B_base = s1;
    const Reg reg_B2 = s2;
    const Reg reg_B3 = s3;
    const Reg reg_bias = s4;
    const Reg reg_M = s5; // M saved for repeated vsetvli configs

    const FReg freg_b0 = fa0;
    const FReg freg_b1 = fa1;
    const FReg freg_b2 = fa2;
    const FReg freg_b3 = fa3;

    const VReg v_c0(0);
    const VReg v_c1(4);
    const VReg v_c2(8);
    const VReg v_c3(12);
    const VReg v_a0(16); // EMUL=m2 at e16: occupies v16-v17
    const VReg v_a1(20); // EMUL=m2 at e16: occupies v20-v21
    const VReg v_tmp(24);

    // sp shrinks by 48 bytes: 6 callee-saved regs × 8 = 48,
    // which keeps sp 16-byte aligned per the RISC-V LP64 ABI.
    addi(sp, sp, -48);
    sd(reg_A_base, sp, 0);
    sd(reg_B_base, sp, 8);
    sd(reg_B2, sp, 16);
    sd(reg_B3, sp, 24);
    sd(reg_bias, sp, 32);
    sd(reg_M, sp, 40);

    ld(reg_A_base, reg_param, 0);
    ld(reg_B_base, reg_param, 8);
    ld(reg_C, reg_param, 16);
    ld(reg_N, reg_param, 24);
    ld(reg_M, reg_param, 32); // keep M around to re-issue vsetvli
    ld(reg_K_val, reg_param, 40);
    lw(reg_beta, reg_param, 48);
    ld(reg_bias, reg_param, 56);

    // Initial vsetvli for C/bias work (e32 m4).
    vsetvli(x0, reg_M, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    li(reg_lda, LDA_bytes);
    li(reg_ldb, LDB_bytes);
    li(reg_ldc, LDC_bytes);

    if (use_single_b) {
        li(s2, N_stride_B);
        li(s3, N_stride_C);
    }

    srli(reg_K_main, reg_K_val, 2);
    slli(reg_K_main, reg_K_main, 2);

    mv(reg_n, x0);

    Label lbl_n_loop, lbl_n_tail, lbl_n_tail_loop, lbl_n_end;

    L(lbl_n_loop);
    addi(reg_tmp0, reg_n, 4);
    blt(reg_N, reg_tmp0, lbl_n_tail);

    mv(reg_A, reg_A_base);

    mv(reg_B0, reg_B_base);
    if (!use_single_b) {
        add(reg_B1, reg_B_base, reg_ldb);
        add(reg_B2, reg_B1, reg_ldb);
        add(reg_B3, reg_B2, reg_ldb);
    }

    // Zero accumulators at e32 m4 (4 phys regs each).
    vmv_v_i(v_c0, 0);
    vmv_v_i(v_c1, 0);
    vmv_v_i(v_c2, 0);
    vmv_v_i(v_c3, 0);

    // Switch to e16/m2 for the K loop. VLMAX matches → vl unchanged.
    vsetvli(x0, reg_M, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);

    mv(reg_k, x0);
    Label lbl_k_main_end, lbl_k_tail, lbl_k_tail_end;

    auto emit_b_load = [&]() {
        flh(freg_b0, reg_B0, 0);
        if (use_single_b) {
            flh(freg_b1, reg_B0, LDB_bytes);
            flh(freg_b2, reg_B0, 2 * LDB_bytes);
            flh(freg_b3, reg_B0, 3 * LDB_bytes);
        } else {
            flh(freg_b1, reg_B1, 0);
            flh(freg_b2, reg_B2, 0);
            flh(freg_b3, reg_B3, 0);
        }
    };

    auto emit_b_advance = [&]() {
        addi(reg_B0, reg_B0, 2);
        if (!use_single_b) {
            addi(reg_B1, reg_B1, 2);
            addi(reg_B2, reg_B2, 2);
            addi(reg_B3, reg_B3, 2);
        }
    };

    auto emit_pipelined_step = [&](const VReg &v_next, const VReg &v_cur) {
        vle16_v(v_next, reg_A);
        add(reg_A, reg_A, reg_lda);
        emit_b_load();
        vfwmacc_vf(v_c0, freg_b0, v_cur);
        vfwmacc_vf(v_c1, freg_b1, v_cur);
        vfwmacc_vf(v_c2, freg_b2, v_cur);
        vfwmacc_vf(v_c3, freg_b3, v_cur);
        emit_b_advance();
    };

    auto emit_drain_step = [&](const VReg &v_cur) {
        emit_b_load();
        vfwmacc_vf(v_c0, freg_b0, v_cur);
        vfwmacc_vf(v_c1, freg_b1, v_cur);
        vfwmacc_vf(v_c2, freg_b2, v_cur);
        vfwmacc_vf(v_c3, freg_b3, v_cur);
        emit_b_advance();
    };

    beq(reg_K_main, x0, lbl_k_tail);

    addi(reg_tmp1, reg_K_main, -4);

    vle16_v(v_a0, reg_A);
    add(reg_A, reg_A, reg_lda);

    align(16);
    {
        Label lbl_pipe, lbl_pipe_end;
        L(lbl_pipe);
        bge(reg_k, reg_tmp1, lbl_pipe_end);

        emit_pipelined_step(v_a1, v_a0);
        emit_pipelined_step(v_a0, v_a1);
        emit_pipelined_step(v_a1, v_a0);
        emit_pipelined_step(v_a0, v_a1);

        addi(reg_k, reg_k, 4);
        j_(lbl_pipe);
        L(lbl_pipe_end);
    }

    emit_pipelined_step(v_a1, v_a0);
    emit_pipelined_step(v_a0, v_a1);
    emit_pipelined_step(v_a1, v_a0);
    emit_drain_step(v_a1);
    addi(reg_k, reg_k, 4);

    L(lbl_k_main_end);

    L(lbl_k_tail);
    bge(reg_k, reg_K_val, lbl_k_tail_end);

    vle16_v(v_a0, reg_A);
    add(reg_A, reg_A, reg_lda);
    emit_b_load();
    vfwmacc_vf(v_c0, freg_b0, v_a0);
    vfwmacc_vf(v_c1, freg_b1, v_a0);
    vfwmacc_vf(v_c2, freg_b2, v_a0);
    vfwmacc_vf(v_c3, freg_b3, v_a0);
    emit_b_advance();
    addi(reg_k, reg_k, 1);
    j_(lbl_k_tail);

    L(lbl_k_tail_end);

    // Switch back to e32 m4 for bias add + C store (f32 ops).
    vsetvli(x0, reg_M, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    {
        Label lbl_no_bias;
        beq(reg_bias, x0, lbl_no_bias);
        vle32_v(v_tmp, reg_bias);
        vfadd_vv(v_c0, v_c0, v_tmp);
        vfadd_vv(v_c1, v_c1, v_tmp);
        vfadd_vv(v_c2, v_c2, v_tmp);
        vfadd_vv(v_c3, v_c3, v_tmp);
        L(lbl_no_bias);
    }

    {
        mv(reg_tmp0, reg_C);
        Label lbl_bz, lbl_store_done;
        beq(reg_beta, x0, lbl_bz);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c0);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c1);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c2);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c3);
        vse32_v(v_tmp, reg_tmp0);

        j_(lbl_store_done);

        L(lbl_bz);
        vse32_v(v_c0, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c1, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c2, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c3, reg_tmp0);

        L(lbl_store_done);
    }

    if (use_single_b) {
        add(reg_B_base, reg_B_base, s2);
        add(reg_C, reg_C, s3);
    } else {
        li(reg_tmp1, N_stride_B);
        add(reg_B_base, reg_B_base, reg_tmp1);
        li(reg_tmp1, N_stride_C);
        add(reg_C, reg_C, reg_tmp1);
    }

    addi(reg_n, reg_n, 4);
    j_(lbl_n_loop);

    // ---- N tail: 1 column at a time ----
    L(lbl_n_tail);

    L(lbl_n_tail_loop);
    bge(reg_n, reg_N, lbl_n_end);

    mv(reg_A, reg_A_base);
    mv(reg_B0, reg_B_base);

    // We're at e32 m4 here (after the previous iter or initial setup).
    vmv_v_i(v_c0, 0);

    // Switch to e16 m2 for the single-column K loop.
    vsetvli(x0, reg_M, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);

    mv(reg_k, x0);
    Label lbl_kt2, lbl_kt2_end;

    L(lbl_kt2);
    bge(reg_k, reg_K_val, lbl_kt2_end);

    vle16_v(v_a0, reg_A);
    flh(freg_b0, reg_B0, 0);
    vfwmacc_vf(v_c0, freg_b0, v_a0);
    add(reg_A, reg_A, reg_lda);
    addi(reg_B0, reg_B0, 2);
    addi(reg_k, reg_k, 1);
    j_(lbl_kt2);

    L(lbl_kt2_end);

    // Back to e32 m4 for bias add + C store.
    vsetvli(x0, reg_M, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    {
        Label lbl_no_bias2;
        beq(reg_bias, x0, lbl_no_bias2);
        vle32_v(v_tmp, reg_bias);
        vfadd_vv(v_c0, v_c0, v_tmp);
        L(lbl_no_bias2);
    }

    {
        Label lbl_bz2, lbl_done2;
        beq(reg_beta, x0, lbl_bz2);
        vle32_v(v_tmp, reg_C);
        vfadd_vv(v_tmp, v_tmp, v_c0);
        vse32_v(v_tmp, reg_C);
        j_(lbl_done2);
        L(lbl_bz2);
        vse32_v(v_c0, reg_C);
        L(lbl_done2);
    }

    add(reg_B_base, reg_B_base, reg_ldb);
    add(reg_C, reg_C, reg_ldc);

    addi(reg_n, reg_n, 1);
    j_(lbl_n_tail_loop);

    L(lbl_n_end);

    ld(reg_A_base, sp, 0);
    ld(reg_B_base, sp, 8);
    ld(reg_B2, sp, 16);
    ld(reg_B3, sp, 24);
    ld(reg_bias, sp, 32);
    ld(reg_M, sp, 40);
    addi(sp, sp, 48);
    ret();
#else
    ret();
#endif
}

brgemm_kernel_f16_t::brgemm_kernel_f16_t(const brgemm_desc_t &brg)
    : brg_(brg), jit_kernel_(new jit_brgemm_f16_kernel_t(brg)) {}

brgemm_kernel_f16_t::~brgemm_kernel_f16_t() {
    delete jit_kernel_;
}

status_t brgemm_kernel_f16_t::create_kernel() {
    return jit_kernel_->create_kernel();
}

void brgemm_kernel_f16_t::operator()(brgemm_kernel_params_t *p) const {
    (*jit_kernel_)(p);
}

// =====================================================================
// bf16 JIT kernel: bf16 x bf16 -> f32 via widening FMA (Zvfbfwma).
// =====================================================================

struct jit_brgemm_bf16_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_bf16_kernel_t)

    jit_brgemm_bf16_kernel_t(const brgemm_desc_t &brg)
        : jit_generator_t("rv64_brgemm_kernel_bf16_jit"), brg_(brg) {}

    void operator()(brgemm_kernel_params_t *p) const {
        jit_generator_t::operator()(p);
    }

    const brgemm_desc_t &get_brg() const { return brg_; }

protected:
    void generate() override;

private:
    brgemm_desc_t brg_;
};

void jit_brgemm_bf16_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const dim_t LDA_bytes = brg_.LDA * brg_.typesize_A; // bf16 → ×2
    const dim_t LDB_bytes = brg_.LDB * brg_.typesize_B; // bf16 → ×2
    const dim_t LDC_bytes = brg_.LDC * brg_.typesize_C; // f32  → ×4
    const dim_t N_stride_B = 4 * LDB_bytes;
    const dim_t N_stride_C = 4 * LDC_bytes;

    // Single-B-pointer optimization: same predicate as f32, but LDB_bytes
    // is half the f32 value for the same LDB, so this is more often true.
    const bool use_single_b = (3 * LDB_bytes <= 2047);

    const Reg reg_param = a0;
    const Reg reg_tmp0 = a0;

    const Reg reg_A = a1;
    const Reg reg_n = a2;
    const Reg reg_C = a3;
    const Reg reg_k = a4;
    const Reg reg_K_main = a5;
    const Reg reg_B0 = a6;
    const Reg reg_B1 = a7;

    const Reg reg_lda = t0;
    const Reg reg_ldb = t1;
    const Reg reg_ldc = t2;
    const Reg reg_K_val = t3;
    const Reg reg_N = t4;
    const Reg reg_beta = t5;
    const Reg reg_tmp1 = t6;

    const Reg reg_A_base = s0;
    const Reg reg_B_base = s1;
    const Reg reg_B2 = s2;
    const Reg reg_B3 = s3;
    const Reg reg_bias = s4;
    const Reg reg_M = s5; // M kept around for repeated vsetvli toggles

    // B scalars are bf16; flh loads 16 bits NaN-boxed, vfwmaccbf16.vf
    // reinterprets the low 16 bits as bf16.
    const FReg freg_b0 = fa0;
    const FReg freg_b1 = fa1;
    const FReg freg_b2 = fa2;
    const FReg freg_b3 = fa3;

    // C at e32/m4 (f32 widening dest); A at e16/m2 (bf16 source).
    // vd kept off v0 to avoid overlap with the mask register.
    const VReg v_c0(4); // LMUL=m4: occupies v4-v7
    const VReg v_c1(8); // v8-v11
    const VReg v_c2(12); // v12-v15
    const VReg v_c3(16); // v16-v19
    const VReg v_a0(20); // EMUL=m2: occupies v20-v21
    const VReg v_a1(24); // EMUL=m2: occupies v24-v25
    const VReg v_tmp(28); // v28-v31

    addi(sp, sp, -56);
    sd(reg_A_base, sp, 0);
    sd(reg_B_base, sp, 8);
    sd(reg_B2, sp, 16);
    sd(reg_B3, sp, 24);
    sd(reg_bias, sp, 32);
    sd(reg_M, sp, 40);

    ld(reg_A_base, reg_param, 0);
    ld(reg_B_base, reg_param, 8);
    ld(reg_C, reg_param, 16);
    ld(reg_N, reg_param, 24);
    ld(reg_M, reg_param, 32); // keep M around for repeated vsetvli toggles
    ld(reg_K_val, reg_param, 40);
    lw(reg_beta, reg_param, 48);
    ld(reg_bias, reg_param, 56);

    // e32/m4 for C/bias; toggle to e16/m2 around the FMA loop because
    // vfwmaccbf16.vf is defined at SEW=e16. LMUL=m2 chosen so VLMAX
    // matches the outer e32/m4 setting and vl is preserved.
    vsetvli(x0, reg_M, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    li(reg_lda, LDA_bytes);
    li(reg_ldb, LDB_bytes);
    li(reg_ldc, LDC_bytes);

    if (use_single_b) {
        li(s2, N_stride_B);
        li(s3, N_stride_C);
    }

    srli(reg_K_main, reg_K_val, 2);
    slli(reg_K_main, reg_K_main, 2);

    mv(reg_n, x0);

    Label lbl_n_loop, lbl_n_tail, lbl_n_tail_loop, lbl_n_end;

    L(lbl_n_loop);
    addi(reg_tmp0, reg_n, 4);
    blt(reg_N, reg_tmp0, lbl_n_tail);

    mv(reg_A, reg_A_base);

    mv(reg_B0, reg_B_base);
    if (!use_single_b) {
        add(reg_B1, reg_B_base, reg_ldb);
        add(reg_B2, reg_B1, reg_ldb);
        add(reg_B3, reg_B2, reg_ldb);
    }

    // Zero accumulators at e32 m4 (each m4 group is 4 phys regs).
    vmv_v_i(v_c0, 0);
    vmv_v_i(v_c1, 0);
    vmv_v_i(v_c2, 0);
    vmv_v_i(v_c3, 0);

    // Switch to e16/m2 for the K loop (vfwmaccbf16.vf requires SEW=e16).
    vsetvli(x0, reg_M, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);

    mv(reg_k, x0);
    Label lbl_k_main_end, lbl_k_tail, lbl_k_tail_end;

    // Load 4 B scalars (bf16 via flh).
    auto emit_b_load = [&]() {
        flh(freg_b0, reg_B0, 0);
        if (use_single_b) {
            flh(freg_b1, reg_B0, LDB_bytes);
            flh(freg_b2, reg_B0, 2 * LDB_bytes);
            flh(freg_b3, reg_B0, 3 * LDB_bytes);
        } else {
            flh(freg_b1, reg_B1, 0);
            flh(freg_b2, reg_B2, 0);
            flh(freg_b3, reg_B3, 0);
        }
    };

    // Advance B pointer(s) by one bf16 element (2 bytes).
    auto emit_b_advance = [&]() {
        addi(reg_B0, reg_B0, 2);
        if (!use_single_b) {
            addi(reg_B1, reg_B1, 2);
            addi(reg_B2, reg_B2, 2);
            addi(reg_B3, reg_B3, 2);
        }
    };

    // Pipelined K step: prefetch next A (bf16, EMUL=m2) while issuing
    // widening FMAs into f32 accumulators using the already-loaded v_cur.
    auto emit_pipelined_step = [&](const VReg &v_next, const VReg &v_cur) {
        vle16_v(v_next, reg_A);
        add(reg_A, reg_A, reg_lda);
        emit_b_load();
        vfwmaccbf16_vf(v_c0, freg_b0, v_cur);
        vfwmaccbf16_vf(v_c1, freg_b1, v_cur);
        vfwmaccbf16_vf(v_c2, freg_b2, v_cur);
        vfwmaccbf16_vf(v_c3, freg_b3, v_cur);
        emit_b_advance();
    };

    // Drain step (final iteration): no A prefetch, avoids OOB read.
    auto emit_drain_step = [&](const VReg &v_cur) {
        emit_b_load();
        vfwmaccbf16_vf(v_c0, freg_b0, v_cur);
        vfwmaccbf16_vf(v_c1, freg_b1, v_cur);
        vfwmaccbf16_vf(v_c2, freg_b2, v_cur);
        vfwmaccbf16_vf(v_c3, freg_b3, v_cur);
        emit_b_advance();
    };

    beq(reg_K_main, x0, lbl_k_tail);

    addi(reg_tmp1, reg_K_main, -4);

    vle16_v(v_a0, reg_A);
    add(reg_A, reg_A, reg_lda);

    align(16);
    {
        Label lbl_pipe, lbl_pipe_end;
        L(lbl_pipe);
        bge(reg_k, reg_tmp1, lbl_pipe_end);

        emit_pipelined_step(v_a1, v_a0);
        emit_pipelined_step(v_a0, v_a1);
        emit_pipelined_step(v_a1, v_a0);
        emit_pipelined_step(v_a0, v_a1);

        addi(reg_k, reg_k, 4);
        j_(lbl_pipe);
        L(lbl_pipe_end);
    }

    emit_pipelined_step(v_a1, v_a0);
    emit_pipelined_step(v_a0, v_a1);
    emit_pipelined_step(v_a1, v_a0);
    emit_drain_step(v_a1);
    addi(reg_k, reg_k, 4);

    L(lbl_k_main_end);

    L(lbl_k_tail);
    bge(reg_k, reg_K_val, lbl_k_tail_end);

    vle16_v(v_a0, reg_A);
    add(reg_A, reg_A, reg_lda);
    emit_b_load();
    vfwmaccbf16_vf(v_c0, freg_b0, v_a0);
    vfwmaccbf16_vf(v_c1, freg_b1, v_a0);
    vfwmaccbf16_vf(v_c2, freg_b2, v_a0);
    vfwmaccbf16_vf(v_c3, freg_b3, v_a0);
    emit_b_advance();
    addi(reg_k, reg_k, 1);
    j_(lbl_k_tail);

    L(lbl_k_tail_end);

    // Switch back to e32/m4 for bias add + C store (f32 ops on m4 accums).
    vsetvli(x0, reg_M, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    // Bias is f32 (same as f32 kernel) → vle32_v + vfadd_vv.
    {
        Label lbl_no_bias;
        beq(reg_bias, x0, lbl_no_bias);
        vle32_v(v_tmp, reg_bias);
        vfadd_vv(v_c0, v_c0, v_tmp);
        vfadd_vv(v_c1, v_c1, v_tmp);
        vfadd_vv(v_c2, v_c2, v_tmp);
        vfadd_vv(v_c3, v_c3, v_tmp);
        L(lbl_no_bias);
    }

    // C is f32 → vle32_v / vse32_v.
    {
        mv(reg_tmp0, reg_C);
        Label lbl_bz, lbl_store_done;
        beq(reg_beta, x0, lbl_bz);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c0);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c1);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c2);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vfadd_vv(v_tmp, v_tmp, v_c3);
        vse32_v(v_tmp, reg_tmp0);

        j_(lbl_store_done);

        L(lbl_bz);
        vse32_v(v_c0, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c1, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c2, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c3, reg_tmp0);

        L(lbl_store_done);
    }

    if (use_single_b) {
        add(reg_B_base, reg_B_base, s2);
        add(reg_C, reg_C, s3);
    } else {
        li(reg_tmp1, N_stride_B);
        add(reg_B_base, reg_B_base, reg_tmp1);
        li(reg_tmp1, N_stride_C);
        add(reg_C, reg_C, reg_tmp1);
    }

    addi(reg_n, reg_n, 4);
    j_(lbl_n_loop);

    // ---- N tail: 1 column at a time ----
    L(lbl_n_tail);

    L(lbl_n_tail_loop);
    bge(reg_n, reg_N, lbl_n_end);

    mv(reg_A, reg_A_base);
    mv(reg_B0, reg_B_base);

    // Currently at e32/m4 (from previous N iteration tail / outer setup).
    vmv_v_i(v_c0, 0);

    // Switch to e16/m2 for single-column FMA loop.
    vsetvli(x0, reg_M, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);

    mv(reg_k, x0);
    Label lbl_kt2, lbl_kt2_end;

    L(lbl_kt2);
    bge(reg_k, reg_K_val, lbl_kt2_end);

    vle16_v(v_a0, reg_A);
    flh(freg_b0, reg_B0, 0);
    vfwmaccbf16_vf(v_c0, freg_b0, v_a0);
    add(reg_A, reg_A, reg_lda);
    addi(reg_B0, reg_B0, 2);
    addi(reg_k, reg_k, 1);
    j_(lbl_kt2);

    L(lbl_kt2_end);

    // Back to e32/m4 for bias + C store.
    vsetvli(x0, reg_M, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    {
        Label lbl_no_bias2;
        beq(reg_bias, x0, lbl_no_bias2);
        vle32_v(v_tmp, reg_bias);
        vfadd_vv(v_c0, v_c0, v_tmp);
        L(lbl_no_bias2);
    }

    {
        Label lbl_bz2, lbl_done2;
        beq(reg_beta, x0, lbl_bz2);
        vle32_v(v_tmp, reg_C);
        vfadd_vv(v_tmp, v_tmp, v_c0);
        vse32_v(v_tmp, reg_C);
        j_(lbl_done2);
        L(lbl_bz2);
        vse32_v(v_c0, reg_C);
        L(lbl_done2);
    }

    add(reg_B_base, reg_B_base, reg_ldb);
    add(reg_C, reg_C, reg_ldc);

    addi(reg_n, reg_n, 1);
    j_(lbl_n_tail_loop);

    L(lbl_n_end);

    ld(reg_A_base, sp, 0);
    ld(reg_B_base, sp, 8);
    ld(reg_B2, sp, 16);
    ld(reg_B3, sp, 24);
    ld(reg_bias, sp, 32);
    ld(reg_M, sp, 40);
    addi(sp, sp, 56);
    ret();
#else
    ret();
#endif
}

brgemm_kernel_bf16_t::brgemm_kernel_bf16_t(const brgemm_desc_t &brg)
    : brg_(brg), jit_kernel_(new jit_brgemm_bf16_kernel_t(brg)) {}

brgemm_kernel_bf16_t::~brgemm_kernel_bf16_t() {
    delete jit_kernel_;
}

status_t brgemm_kernel_bf16_t::create_kernel() {
    return jit_kernel_->create_kernel();
}

void brgemm_kernel_bf16_t::operator()(brgemm_kernel_params_t *p) const {
    (*jit_kernel_)(p);
}

// =====================================================================
// int8 JIT kernel: s8 × s8 → s32.
// A is pre-widened to s32 during packing and loaded here as e32/m4; B is
// loaded one byte at a time with `lb` and accumulated with vmacc.vx.
// u8 / mixed-sign combos are rejected at brgemm_desc_init.
// =====================================================================

struct jit_brgemm_s8_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_s8_kernel_t)

    jit_brgemm_s8_kernel_t(const brgemm_desc_t &brg)
        : jit_generator_t("rv64_brgemm_kernel_s8_jit"), brg_(brg) {}

    void operator()(brgemm_kernel_params_t *p) const {
        jit_generator_t::operator()(p);
    }

    const brgemm_desc_t &get_brg() const { return brg_; }

protected:
    void generate() override;

private:
    brgemm_desc_t brg_;
};

void jit_brgemm_s8_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    using namespace Xbyak_riscv;

    const dim_t s32_size
            = static_cast<dim_t>(types::data_type_size(data_type::s32));
    const dim_t LDA_bytes = brg_.LDA * s32_size;
    const dim_t LDB_bytes = brg_.LDB * brg_.typesize_B;
    const dim_t LDC_bytes = brg_.LDC * brg_.typesize_C;
    const dim_t N_stride_B = 4 * LDB_bytes;
    const dim_t N_stride_C = 4 * LDC_bytes;

    const bool use_single_b = (3 * LDB_bytes <= 2047);

    const Reg reg_param = a0;
    const Reg reg_tmp0 = a0;

    const Reg reg_A = a1;
    const Reg reg_n = a2;
    const Reg reg_C = a3;
    const Reg reg_k = a4;
    const Reg reg_K_main = a5;
    const Reg reg_B0 = a6;
    const Reg reg_B1 = a7;

    const Reg reg_lda = t0;
    const Reg reg_ldb = t1;
    const Reg reg_ldc = t2;
    const Reg reg_K_val = t3;
    const Reg reg_N = t4;
    const Reg reg_beta = t5;
    const Reg reg_tmp1 = t6;

    const Reg reg_A_base = s0;
    const Reg reg_B_base = s1;

    const Reg reg_B2 = s2;
    const Reg reg_B3 = s3;
    const Reg reg_n_stride_b = s2;
    const Reg reg_n_stride_c = s3;

    // Register layout (LMUL=m4 throughout, 28/32 phys regs, v0 free for mask):
    //   v4-v7   v_c0   accumulator col 0
    //   v8-v11  v_c1   accumulator col 1
    //   v12-v15 v_c2   accumulator col 2
    //   v16-v19 v_c3   accumulator col 3
    //   v20-v23 v_a0   A double-buffer 0
    //   v24-v27 v_a1   A double-buffer 1
    //   v28-v31 v_tmp  scratch for C load/store
    const VReg v_c0(4);
    const VReg v_c1(8);
    const VReg v_c2(12);
    const VReg v_c3(16);
    const VReg v_a0(20);
    const VReg v_a1(24);
    const VReg v_tmp(28);

    addi(sp, sp, -32);
    sd(reg_A_base, sp, 0);
    sd(reg_B_base, sp, 8);
    sd(reg_B2, sp, 16);
    sd(reg_B3, sp, 24);

    ld(reg_A_base, reg_param, 0);
    ld(reg_B_base, reg_param, 8);
    ld(reg_C, reg_param, 16);
    ld(reg_N, reg_param, 24);
    ld(reg_tmp1, reg_param, 32); // M (one-shot, for vsetvli)
    ld(reg_K_val, reg_param, 40);
    lw(reg_beta, reg_param, 48);

    vsetvli(x0, reg_tmp1, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    li(reg_lda, LDA_bytes);
    li(reg_ldb, LDB_bytes);
    li(reg_ldc, LDC_bytes);

    if (use_single_b) {
        li(reg_n_stride_b, N_stride_B);
        li(reg_n_stride_c, N_stride_C);
    }

    // K_main = (K / 4) * 4 — the 4× unrolled portion of the K loop.
    srli(reg_K_main, reg_K_val, 2);
    slli(reg_K_main, reg_K_main, 2);

    // ---- N outer loop: 4 columns per iteration ----
    mv(reg_n, x0);

    Label lbl_n_loop, lbl_n_tail, lbl_n_tail_loop, lbl_n_end;

    // B-scalar FMA: load one signed byte from reg_b+imm_off into a GPR, then
    // vmacc.vx into the named accumulator against the supplied A row.
    auto emit_b_fma = [&](const Reg &reg_b, int imm_off, const VReg &v_c,
                              const VReg &v_a_cur) {
        lb(reg_tmp0, reg_b, imm_off);
        vmacc_vx(v_c, reg_tmp0, v_a_cur);
    };

    auto emit_b_advance = [&]() {
        addi(reg_B0, reg_B0, 1);
        if (!use_single_b) {
            addi(reg_B1, reg_B1, 1);
            addi(reg_B2, reg_B2, 1);
            addi(reg_B3, reg_B3, 1);
        }
    };

    L(lbl_n_loop);
    addi(reg_tmp0, reg_n, 4);
    blt(reg_N, reg_tmp0, lbl_n_tail);

    mv(reg_A, reg_A_base);
    mv(reg_B0, reg_B_base);
    if (!use_single_b) {
        add(reg_B1, reg_B_base, reg_ldb);
        add(reg_B2, reg_B1, reg_ldb);
        add(reg_B3, reg_B2, reg_ldb);
    }

    vmv_v_i(v_c0, 0);
    vmv_v_i(v_c1, 0);
    vmv_v_i(v_c2, 0);
    vmv_v_i(v_c3, 0);

    // ---- K main loop (4× unrolled, double-buffered A loads) ----
    mv(reg_k, x0);
    Label lbl_k_tail, lbl_k_tail_end;

    auto emit_fma4 = [&](const VReg &v_a_cur) {
        if (use_single_b) {
            emit_b_fma(reg_B0, 0, v_c0, v_a_cur);
            emit_b_fma(reg_B0, (int)LDB_bytes, v_c1, v_a_cur);
            emit_b_fma(reg_B0, (int)(2 * LDB_bytes), v_c2, v_a_cur);
            emit_b_fma(reg_B0, (int)(3 * LDB_bytes), v_c3, v_a_cur);
        } else {
            emit_b_fma(reg_B0, 0, v_c0, v_a_cur);
            emit_b_fma(reg_B1, 0, v_c1, v_a_cur);
            emit_b_fma(reg_B2, 0, v_c2, v_a_cur);
            emit_b_fma(reg_B3, 0, v_c3, v_a_cur);
        }
    };

    // Pipelined step: prefetch next A into v_next while accumulating with
    // the already-loaded v_cur.
    auto emit_pipelined_step = [&](const VReg &v_next, const VReg &v_cur) {
        vle32_v(v_next, reg_A);
        add(reg_A, reg_A, reg_lda);
        emit_fma4(v_cur);
        emit_b_advance();
    };

    // Drain step: FMA only, no prefetch (avoids OOB A read).
    auto emit_drain_step = [&](const VReg &v_cur) {
        emit_fma4(v_cur);
        emit_b_advance();
    };

    beq(reg_K_main, x0, lbl_k_tail);

    // K_pipe = K_main - 4: loop bound for the fully-pipelined phase.
    addi(reg_tmp1, reg_K_main, -4);

    vle32_v(v_a0, reg_A); // preload A[0]
    add(reg_A, reg_A, reg_lda);

    align(16);
    {
        Label lbl_pipe, lbl_pipe_end;
        L(lbl_pipe);
        bge(reg_k, reg_tmp1, lbl_pipe_end);

        emit_pipelined_step(v_a1, v_a0);
        emit_pipelined_step(v_a0, v_a1);
        emit_pipelined_step(v_a1, v_a0);
        emit_pipelined_step(v_a0, v_a1);

        addi(reg_k, reg_k, 4);
        j_(lbl_pipe);
        L(lbl_pipe_end);
    }

    // Remainder: 4 drain-equivalent steps covering A[K_pipe..K_main-1].
    emit_pipelined_step(v_a1, v_a0);
    emit_pipelined_step(v_a0, v_a1);
    emit_pipelined_step(v_a1, v_a0);
    emit_drain_step(v_a1);
    addi(reg_k, reg_k, 4);

    // ---- K tail (1 step at a time) ----
    L(lbl_k_tail);
    bge(reg_k, reg_K_val, lbl_k_tail_end);

    vle32_v(v_a0, reg_A);
    add(reg_A, reg_A, reg_lda);
    emit_fma4(v_a0);
    emit_b_advance();
    addi(reg_k, reg_k, 1);
    j_(lbl_k_tail);

    L(lbl_k_tail_end);

    // ---- Store accumulators to C, honoring beta ----
    {
        mv(reg_tmp0, reg_C);
        Label lbl_bz, lbl_store_done;
        beq(reg_beta, x0, lbl_bz);

        // beta != 0: C[col] = C[col] + accum
        vle32_v(v_tmp, reg_tmp0);
        vadd_vv(v_tmp, v_tmp, v_c0);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vadd_vv(v_tmp, v_tmp, v_c1);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vadd_vv(v_tmp, v_tmp, v_c2);
        vse32_v(v_tmp, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);

        vle32_v(v_tmp, reg_tmp0);
        vadd_vv(v_tmp, v_tmp, v_c3);
        vse32_v(v_tmp, reg_tmp0);

        j_(lbl_store_done);

        // beta == 0: C[col] = accum (overwrite)
        L(lbl_bz);
        vse32_v(v_c0, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c1, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c2, reg_tmp0);
        add(reg_tmp0, reg_tmp0, reg_ldc);
        vse32_v(v_c3, reg_tmp0);

        L(lbl_store_done);
    }

    if (use_single_b) {
        add(reg_B_base, reg_B_base, reg_n_stride_b);
        add(reg_C, reg_C, reg_n_stride_c);
    } else {
        li(reg_tmp1, N_stride_B);
        add(reg_B_base, reg_B_base, reg_tmp1);
        li(reg_tmp1, N_stride_C);
        add(reg_C, reg_C, reg_tmp1);
    }

    addi(reg_n, reg_n, 4);
    j_(lbl_n_loop);

    // ---- N tail: 1 column at a time ----
    L(lbl_n_tail);

    L(lbl_n_tail_loop);
    bge(reg_n, reg_N, lbl_n_end);

    mv(reg_A, reg_A_base);
    mv(reg_B0, reg_B_base);

    vmv_v_i(v_c0, 0);

    mv(reg_k, x0);
    Label lbl_kt2, lbl_kt2_end;

    L(lbl_kt2);
    bge(reg_k, reg_K_val, lbl_kt2_end);

    vle32_v(v_a0, reg_A);
    add(reg_A, reg_A, reg_lda);
    emit_b_fma(reg_B0, 0, v_c0, v_a0);
    addi(reg_B0, reg_B0, 1);
    addi(reg_k, reg_k, 1);
    j_(lbl_kt2);

    L(lbl_kt2_end);

    {
        Label lbl_bz2, lbl_done2;
        beq(reg_beta, x0, lbl_bz2);
        vle32_v(v_tmp, reg_C);
        vadd_vv(v_tmp, v_tmp, v_c0);
        vse32_v(v_tmp, reg_C);
        j_(lbl_done2);
        L(lbl_bz2);
        vse32_v(v_c0, reg_C);
        L(lbl_done2);
    }

    add(reg_B_base, reg_B_base, reg_ldb);
    add(reg_C, reg_C, reg_ldc);

    addi(reg_n, reg_n, 1);
    j_(lbl_n_tail_loop);

    L(lbl_n_end);

    ld(reg_A_base, sp, 0);
    ld(reg_B_base, sp, 8);
    ld(reg_B2, sp, 16);
    ld(reg_B3, sp, 24);
    addi(sp, sp, 32);
    ret();
#else
    ret();
#endif
}

brgemm_kernel_s8_t::brgemm_kernel_s8_t(const brgemm_desc_t &brg)
    : brg_(brg), jit_kernel_(new jit_brgemm_s8_kernel_t(brg)) {}

brgemm_kernel_s8_t::~brgemm_kernel_s8_t() {
    delete jit_kernel_;
}

status_t brgemm_kernel_s8_t::create_kernel() {
    return jit_kernel_->create_kernel();
}

void brgemm_kernel_s8_t::operator()(brgemm_kernel_params_t *p) const {
    (*jit_kernel_)(p);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
