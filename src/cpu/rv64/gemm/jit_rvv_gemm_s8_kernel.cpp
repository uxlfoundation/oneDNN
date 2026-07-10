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

#include "cpu/rv64/gemm/jit_rvv_gemm_s8_kernel.hpp"

#include <array>
#include <memory>
#include <mutex>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

using namespace Xbyak_riscv;

jit_rvv_gemm_s8_kernel_t::jit_rvv_gemm_s8_kernel_t(dim_t n_cols, bool isTransA,
        bool isTransB, bool b_signed, bool dst_is_f32, bool has_bias)
    : jit_generator_t("rv64_gemm_kernel_s8_jit")
    , n_cols_(n_cols)
    , isTransA_(isTransA)
    , isTransB_(isTransB)
    , b_signed_(b_signed)
    , dst_is_f32_(dst_is_f32)
    , has_bias_(has_bias) {
    create_kernel();
}

void jit_rvv_gemm_s8_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;

    const Reg reg_A_ptr = a1; // running pointer into A (weights)
    const Reg reg_m = a2; // tile height (used for vsetvli)
    const Reg reg_C_base = a3; // base pointer to C(:, 0)

    const Reg reg_lda_bytes = t0; // A is s8: 1 byte per element
    const Reg reg_ldb_bytes = t1; // B is s8/u8: 1 byte per element
    const Reg reg_ldc_bytes = t2; // C is s32/f32: 4 bytes per element
    const Reg reg_K = t3;
    const Reg reg_alpha_bits = t4;
    const Reg reg_bias_ptr = t4; // reuse after alpha bits moved to freg
    const Reg reg_beta_bits = t5;

    const Reg reg_k = a4; // current k counter
    const Reg reg_B0_ptr = a6; // running pointer into B
    const Reg reg_tmp0 = a7;
    const FReg freg_alpha = fa0;
    const FReg freg_beta = fa1;
    // B scalars are kept in GPRs across the per-K vwmacc[vx|su.vx] calls.
    const Reg reg_b[6] = {a5, t6, s2, s3, s4, s5};

    const VReg v_c[6]
            = {VReg(0), VReg(4), VReg(8), VReg(12), VReg(16), VReg(20)};
    // K-loop temporaries.
    const VReg v_a_e16(24); // e16 LMUL=m2 sign-/zero-extended A
    const VReg v_a_e8(26); // e8 LMUL=m1 source row
    // C-update temporaries.
    const VReg v_tmp(24); // e32 LMUL=m4 scratch / C-load temporary
    const VReg v_bias(28); // e32 LMUL=m4 bias loader (f32 dst path)

    // Layout of call_params_t (offsets in bytes):
    //   0  : const void *A
    //   8  : const void *B
    //   16 : void *C
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

    // A is s8 (1 byte), B is s8/u8 (1 byte): element stride == byte stride.
    // C is s32/f32 (4 bytes): ldc * 4 = byte stride.
    slli(reg_ldc_bytes, reg_ldc_bytes, 2);

    // Save callee-saved registers (s2..s5 hold reg_b[2..5] when n_cols >= 3).
    const bool need_callee_save = (n_cols_ >= 3);
    if (need_callee_save) {
        addi(sp, sp, -32);
        sd(s2, sp, 0);
        sd(s3, sp, 8);
        sd(s4, sp, 16);
        sd(s5, sp, 24);
    }

    // Initialize accumulators.
    vsetvli(x0, reg_m, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    for (dim_t c = 0; c < n_cols_; c++)
        vmv_v_i(v_c[c], 0);

    auto emit_load_a_e8 = [&]() {
        if (isTransA_) {
            vlse8_v(v_a_e8, reg_A_ptr, reg_lda_bytes);
        } else {
            vle8_v(v_a_e8, reg_A_ptr);
        }
    };

    auto emit_extend_a = [&]() {
        // Sign-extend v_a_e8 (e8 LMUL=m1) to v_a_e16 (e16 LMUL=m2).
        li(reg_tmp0, 1);
        vwmul_vx(v_a_e16, v_a_e8, reg_tmp0);
    };

    auto emit_load_b_consecutive = [&]() {
        // isTransB_: B[k, 0..n_cols-1] are n_cols consecutive bytes.
        for (dim_t c = 0; c < n_cols_; c++) {
            if (b_signed_) {
                lb(reg_b[c], reg_B0_ptr, static_cast<int32_t>(c));
            } else {
                lbu(reg_b[c], reg_B0_ptr, static_cast<int32_t>(c));
            }
        }
    };

    auto emit_load_b_strided = [&]() {
        // !isTransB_: B[k, c] = mem[reg_B0_ptr + c * ldb_bytes].
        auto lb_or_lbu = [&](const Reg &dst, const Reg &base, int32_t off) {
            if (b_signed_)
                lb(dst, base, off);
            else
                lbu(dst, base, off);
        };
        lb_or_lbu(reg_b[0], reg_B0_ptr, 0);
        if (n_cols_ > 1) {
            add(reg_tmp0, reg_B0_ptr, reg_ldb_bytes); // &B[k, 1]
            for (dim_t c = 1; c < n_cols_; c++) {
                lb_or_lbu(reg_b[c], reg_tmp0, 0);
                if (c + 1 < n_cols_) add(reg_tmp0, reg_tmp0, reg_ldb_bytes);
            }
        }
    };

    auto emit_compute = [&]() {
        // SEW=e16 LMUL=m2 here. v_c[c] is e32 LMUL=m4 (2*SEW).
        for (dim_t c = 0; c < n_cols_; c++) {
            vwmacc_vx(v_c[c], reg_b[c], v_a_e16);
        }
    };

    auto emit_advance_a = [&]() {
        if (isTransA_) {
            addi(reg_A_ptr, reg_A_ptr, 1);
        } else {
            add(reg_A_ptr, reg_A_ptr, reg_lda_bytes);
        }
    };

    auto emit_advance_b = [&]() {
        if (isTransB_) {
            add(reg_B0_ptr, reg_B0_ptr, reg_ldb_bytes);
        } else {
            addi(reg_B0_ptr, reg_B0_ptr, 1);
        }
    };

    // Main K loop. Each iteration: load A (e8) -> extend to e16 -> set SEW=e16
    // -> load B scalars -> vwmacc into accumulators -> advance A/B.
    Label label_k_done;
    Label label_loop_k;

    beqz(reg_K, label_k_done);
    mv(reg_k, x0);

    L(label_loop_k);

    vsetvli(x0, reg_m, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
    emit_load_a_e8();
    emit_extend_a();

    vsetvli(x0, reg_m, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);

    if (isTransB_) {
        emit_load_b_consecutive();
    } else {
        emit_load_b_strided();
    }
    emit_compute();

    emit_advance_a();
    emit_advance_b();

    addi(reg_k, reg_k, 1);
    blt(reg_k, reg_K, label_loop_k);

    L(label_k_done);

    // C-update: switch back to e32 LMUL=m4 (VL=m) for the epilogue.
    vsetvli(x0, reg_m, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

    // Computes the address of C's col_idx column into reg_tmp3.
    auto emit_col_addr = [&](dim_t col_idx) {
        if (col_idx == 0) {
            mv(reg_tmp0, reg_C_base);
        } else {
            li(reg_tmp0, col_idx);
            mul(reg_tmp0, reg_ldc_bytes, reg_tmp0);
            add(reg_tmp0, reg_C_base, reg_tmp0);
        }
    };

    // Preload the per-M bias vector once (broadcast across N columns).
    if (has_bias_) {
        Label label_bias_loaded;
        beqz(reg_bias_ptr, label_bias_loaded);
        vle32_v(v_bias, reg_bias_ptr);
        if (!dst_is_f32_) { vfcvt_rtz_x_f_v(v_bias, v_bias); }
        L(label_bias_loaded);
    }

    if (dst_is_f32_) {
        // f32 dst path: C[col_idx] = alpha * fcvt(acc) + beta * C + bias.
        for (dim_t c = 0; c < n_cols_; c++) {
            Label label_beta_zero, label_skip_bias, label_store, label_done;

            emit_col_addr(c);

            // s32 acc -> f32 in place.
            vfcvt_f_x_v(v_c[c], v_c[c]);

            beqz(reg_beta_bits, label_beta_zero);

            vle32_v(v_tmp, reg_tmp0);
            vfmul_vf(v_tmp, v_tmp, freg_beta);
            vfmul_vf(v_c[c], v_c[c], freg_alpha);
            vfadd_vv(v_tmp, v_tmp, v_c[c]);

            if (has_bias_) {
                beqz(reg_bias_ptr, label_skip_bias);
                vfadd_vv(v_tmp, v_tmp, v_bias);
                L(label_skip_bias);
            }

            vse32_v(v_tmp, reg_tmp0);
            j_(label_done);

            L(label_beta_zero);
            vfmul_vf(v_c[c], v_c[c], freg_alpha);

            if (has_bias_) {
                beqz(reg_bias_ptr, label_store);
                vfadd_vv(v_c[c], v_c[c], v_bias);
            }

            L(label_store);
            vse32_v(v_c[c], reg_tmp0);

            L(label_done);
        }
    } else {
        // s32 dst path.
        for (dim_t c = 0; c < n_cols_; c++) {
            Label label_after_beta, label_skip_bias;

            emit_col_addr(c);

            beqz(reg_beta_bits, label_after_beta);
            // beta != 0: read C and add to accumulator in place.
            vle32_v(v_tmp, reg_tmp0);
            vsadd_vv(v_c[c], v_tmp, v_c[c]);
            L(label_after_beta);

            if (has_bias_) {
                beqz(reg_bias_ptr, label_skip_bias);
                vsadd_vv(v_c[c], v_c[c], v_bias);
                L(label_skip_bias);
            }
            vse32_v(v_c[c], reg_tmp0);
        }
    }

    // Restore callee-saved registers and return.
    if (need_callee_save) {
        ld(s2, sp, 0);
        ld(s3, sp, 8);
        ld(s4, sp, 16);
        ld(s5, sp, 24);
        addi(sp, sp, 32);
    }

    ret();
#else
    ret();
#endif
}

namespace {

template <bool isTransA, bool isTransB, bool b_signed, bool dst_is_f32>
struct jit_rvv_gemm_s8_kernel_storage_t {
    std::array<std::unique_ptr<jit_rvv_gemm_s8_kernel_t>, 8> nb;
    std::array<std::unique_ptr<jit_rvv_gemm_s8_kernel_t>, 8> b;
    jit_rvv_gemm_s8_kernel_table_t table;
};

template <bool isTransA, bool isTransB, bool b_signed, bool dst_is_f32>
jit_rvv_gemm_s8_kernel_storage_t<isTransA, isTransB, b_signed, dst_is_f32> &
get_jit_rvv_gemm_s8_kernel_storage() {
    static jit_rvv_gemm_s8_kernel_storage_t<isTransA, isTransB, b_signed,
            dst_is_f32>
            storage;
    static std::once_flag initialized;

    std::call_once(initialized, [] {
        for (dim_t n_cols = 1; n_cols <= 6; n_cols++) {
            storage.nb[n_cols].reset(new jit_rvv_gemm_s8_kernel_t(
                    n_cols, isTransA, isTransB, b_signed, dst_is_f32, false));
            storage.b[n_cols].reset(new jit_rvv_gemm_s8_kernel_t(
                    n_cols, isTransA, isTransB, b_signed, dst_is_f32, true));
            storage.table.nb[n_cols] = storage.nb[n_cols].get();
            storage.table.b[n_cols] = storage.b[n_cols].get();
        }
    });

    return storage;
}

} // namespace

const jit_rvv_gemm_s8_kernel_table_t &get_jit_rvv_gemm_s8_kernel_table(
        bool isTransA, bool isTransB, bool b_signed, bool dst_is_f32) {
#define DISPATCH(SA, SB, BS, DF) \
    if (isTransA == (SA) && isTransB == (SB) && b_signed == (BS) \
            && dst_is_f32 == (DF)) \
        return get_jit_rvv_gemm_s8_kernel_storage<SA, SB, BS, DF>().table;
    DISPATCH(false, false, true, false)
    DISPATCH(false, false, true, true)
    DISPATCH(false, false, false, false)
    DISPATCH(false, false, false, true)
    DISPATCH(false, true, true, false)
    DISPATCH(false, true, true, true)
    DISPATCH(false, true, false, false)
    DISPATCH(false, true, false, true)
    DISPATCH(true, false, true, false)
    DISPATCH(true, false, true, true)
    DISPATCH(true, false, false, false)
    DISPATCH(true, false, false, true)
    DISPATCH(true, true, true, false)
    DISPATCH(true, true, true, true)
    DISPATCH(true, true, false, false)
    DISPATCH(true, true, false, true)
#undef DISPATCH
    // Unreachable: all 16 combinations are covered above.
    return get_jit_rvv_gemm_s8_kernel_storage<false, false, true, false>()
            .table;
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
