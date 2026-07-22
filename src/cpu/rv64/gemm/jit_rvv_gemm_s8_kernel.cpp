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
#include <cstdlib>
#include <memory>
#include <mutex>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

using namespace Xbyak_riscv;
using namespace dnnl::impl::utils;

jit_rvv_gemm_s8_kernel_t::jit_rvv_gemm_s8_kernel_t(dim_t n_cols, bool isTransA,
        bool isTransB, bool b_signed, data_type_t dst_dt, bool has_bias)
    : jit_generator_t("rv64_gemm_kernel_s8_jit")
    , n_cols_(n_cols)
    , isTransA_(isTransA)
    , isTransB_(isTransB)
    , b_signed_(b_signed)
    , dst_dt_(dst_dt)
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
    const Reg reg_ldc_bytes = t2; // C element size in bytes
    const Reg reg_K = t3;
    const Reg reg_alpha_bits = t4;
    const Reg reg_bias_ptr = t4; // reuse after alpha bits moved to freg
    const Reg reg_beta_bits = t5;

    const Reg reg_k = a4; // current k counter
    const Reg reg_B0_ptr = a6; // running pointer into B
    const Reg reg_tmp0 = a7;
    const FReg freg_alpha = fa0;
    const FReg freg_beta = fa1;
    const FReg freg_bias = fa2; // scalar bias splat (has_bias_ + scalar)
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
    //   72 : const float *bias       (only used when has_bias_)
    //   80 : int bias_is_scalar      (only used when has_bias_)
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

    // Scale ldc from element-count to byte stride.
    if (one_of(dst_dt_, data_type::s8, data_type::u8)) {
        // 1-byte element: element units == byte units, no shift needed.
    } else if (one_of(dst_dt_, data_type::f16, data_type::bf16)) {
        slli(reg_ldc_bytes, reg_ldc_bytes, 1);
    } else {
        slli(reg_ldc_bytes, reg_ldc_bytes, 2);
    }

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

    // Computes the address of C's col_idx column into reg_tmp0.
    auto emit_col_addr = [&](dim_t col_idx) {
        if (col_idx == 0) {
            mv(reg_tmp0, reg_C_base);
        } else {
            li(reg_tmp0, col_idx);
            mul(reg_tmp0, reg_ldc_bytes, reg_tmp0);
            add(reg_tmp0, reg_C_base, reg_tmp0);
        }
    };

    // Preload the per-M bias vector once (broadcast across N columns). A scalar
    // (one-element) bias must be splat from a single float: a full-vector
    // vle32 would read past the one-element object. A per-N bias is a
    // contiguous m-wide load.
    if (has_bias_) {
        Label label_bias_loaded, label_bias_per_element, label_bias_done;
        beqz(reg_bias_ptr, label_bias_loaded);
        lw(reg_tmp0, reg_param, 80);
        // 0 → per-element bias (bias_is_scalar == false): do a contiguous
        // vle32_v load of the m-wide vector. nonzero → scalar bias
        // (bias_is_scalar == true): splat the single float across all m
        // lanes via vfmv.v.f. A full vle32_v on a one-element object would
        // overrun the allocation.
        beqz(reg_tmp0, label_bias_per_element);
        flw(freg_bias, reg_bias_ptr, 0);
        vfmv_v_f(v_bias, freg_bias);
        j_(label_bias_done);
        L(label_bias_per_element);
        vle32_v(v_bias, reg_bias_ptr);
        L(label_bias_done);
        // f32 -> s32 using the rounding mode in fcsr (RNE by default), matching
        // the reference, which adds the f32 bias and rounds via nearbyintf.
        // RTZ was wrong: it truncated 0.75f to 0 instead of 1.
        if (dst_dt_ == data_type::s32) { vfcvt_x_f_v(v_bias, v_bias); }
        L(label_bias_loaded);
    }

    if (one_of(dst_dt_, data_type::f32, data_type::f16, data_type::bf16)) {
        // f32-compatible epilogue: s32 acc -> f32 -> alpha/beta/bias ->
        // store at element width (4 bytes for f32, 2 bytes for f16/bf16).
        // For f16/bf16 we narrow via vfncvt before storing.
        const bool is_f32_dst = (dst_dt_ == data_type::f32);

        for (dim_t c = 0; c < n_cols_; c++) {
            Label label_beta_zero, label_skip_bias, label_store, label_done;

            emit_col_addr(c);

            // s32 acc -> f32 in place.
            vfcvt_f_x_v(v_c[c], v_c[c]);

            beqz(reg_beta_bits, label_beta_zero);

            // Read existing C (element-width-aware: e32 for f32, e16 for f16/bf16).
            // For f16/bf16 we scalar-load the 2-byte value and broadcast to
            // all m lanes via vfmv.v.f. Using a scalar load avoids depending
            // on vector widening (vfwcvt.f.f.v)
            const FReg freg_loaded = fa5;
            if (is_f32_dst) {
                vle32_v(v_tmp, reg_tmp0);
            } else {
                lh(reg_k, reg_tmp0, 0); // sign-extend s16 -> s32 in reg_k
                fmv_w_x(freg_loaded, reg_k);
                vfmv_v_f(v_tmp, freg_loaded);
            }
            vfmul_vf(v_tmp, v_tmp, freg_beta);
            vfmul_vf(v_c[c], v_c[c], freg_alpha);
            vfadd_vv(v_tmp, v_tmp, v_c[c]);

            if (has_bias_) {
                beqz(reg_bias_ptr, label_skip_bias);
                vfadd_vv(v_tmp, v_tmp, v_bias);
                L(label_skip_bias);
            }

            if (!is_f32_dst) {
                vsetvli(x0, reg_m, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
                if (dst_dt_ == data_type::bf16) {
                    vfncvtbf16_f_f_w(v_tmp, v_tmp);
                } else {
                    vfncvt_f_f_w(v_tmp, v_tmp);
                }
                vse16_v(v_tmp, reg_tmp0);
                vsetvli(x0, reg_m, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
            } else {
                vse32_v(v_tmp, reg_tmp0);
            }
            j_(label_done);

            L(label_beta_zero);
            vfmul_vf(v_c[c], v_c[c], freg_alpha);

            if (has_bias_) {
                beqz(reg_bias_ptr, label_store);
                vfadd_vv(v_c[c], v_c[c], v_bias);
            }

            L(label_store);
            if (!is_f32_dst) {
                vsetvli(x0, reg_m, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
                if (dst_dt_ == data_type::bf16) {
                    vfncvtbf16_f_f_w(v_c[c], v_c[c]);
                } else {
                    vfncvt_f_f_w(v_c[c], v_c[c]);
                }
                vse16_v(v_c[c], reg_tmp0);
                vsetvli(x0, reg_m, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
            } else {
                vse32_v(v_c[c], reg_tmp0);
            }

            L(label_done);
        }
    } else if (one_of(dst_dt_, data_type::s8, data_type::u8)) {
        // Narrow 8-bit dst: s32 acc -> f32 -> +bias (if any) -> saturate to
        // [lbound, ubound] -> f32 -> s32 (RNE per fcsr) -> narrow to e8 ->
        // vse8. Alpha is ignored (the driver rejects alpha != 1 for s8/u8
        // dst). Beta must be 0 (overwrite-only): the driver asserts beta == 0
        // and the kernel branch overwrites v_c[c] regardless.
        const bool dst_is_u8 = (dst_dt_ == data_type::u8);

        // Splat scalar saturate bounds into fregs once (reused per column).
        const FReg freg_sat_ubound = fa3;
        const FReg freg_sat_lbound = fa4;
        if (dst_is_u8) {
            li(reg_tmp0, 0x437F0000); // 255.0f
            fmv_w_x(freg_sat_ubound, reg_tmp0);
            li(reg_tmp0, 0x00000000); // 0.0f
            fmv_w_x(freg_sat_lbound, reg_tmp0);
        } else {
            li(reg_tmp0, 0x42FE0000); // 127.0f
            fmv_w_x(freg_sat_ubound, reg_tmp0);
            li(reg_tmp0, 0xC3000000); // -128.0f
            fmv_w_x(freg_sat_lbound, reg_tmp0);
        }

        for (dim_t c = 0; c < n_cols_; c++) {
            emit_col_addr(c);

            // s32 acc -> f32 in place (in v_c[c]).
            vfcvt_f_x_v(v_c[c], v_c[c]);

            // Add bias if requested.
            if (has_bias_) {
                Label label_skip_bias;
                beqz(reg_bias_ptr, label_skip_bias);
                vfadd_vv(v_c[c], v_c[c], v_bias);
                L(label_skip_bias);
            }

            // Saturate to the dst range.
            vfmax_vf(v_c[c], v_c[c], freg_sat_lbound);
            vfmin_vf(v_c[c], v_c[c], freg_sat_ubound);

            // f32 -> s32 / u32 with RNE rounding (fcsr default).
            if (dst_is_u8) {
                vfcvt_xu_f_v(v_c[c], v_c[c]);
            } else {
                vfcvt_x_f_v(v_c[c], v_c[c]);
            }

            // Narrow the s32/u32 accumulator in v_c[c] to e8 and store.
            vsetvli(x0, reg_m, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
            vnsrl_wi(v_tmp, v_c[c], 0);
            vsetvli(x0, reg_m, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
            vnsrl_wi(v_tmp, v_tmp, 0);
            vse8_v(v_tmp, reg_tmp0);
            // Restore e32/m4 state for the next column (or for callee-save
            // epilogue / return).
            vsetvli(x0, reg_m, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        }
    } else {
        // s32 dst path (default branch when dst_dt_ == data_type::s32).
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

// Number of destination data types supported by the s8 GEMM JIT kernel.
// Used as a bound on the storage arrays. Keep in lockstep with
// `supported_dst_dts()` below — adding a new dst means appending both.
constexpr int kNumDstKinds = 6;

// Single source of truth for the supported-dst enumeration. Used to build
// the storage in dst → index order and to answer the reverse lookup in
// `dst_kind_index()`. Adding a new dst is a one-line change here.
const std::array<data_type_t, kNumDstKinds> &supported_dst_dts() {
    static const std::array<data_type_t, kNumDstKinds> dts
            = {data_type::s32, data_type::f32, data_type::s8, data_type::u8,
                    data_type::f16, data_type::bf16};
    return dts;
}

// Map a destination data type to its [0, kNumDstKinds) index in
// `supported_dst_dts()`. Returns -1 for unsupported data types. The caller
// (the dispatcher) checks the return and aborts on -1; assert-only
// validation here would let a missing case silently produce a wrong
// answer in release builds.
int dst_kind_index(data_type_t dst_dt) {
    const auto &dts = supported_dst_dts();
    for (int i = 0; i < kNumDstKinds; ++i) {
        if (dts[i] == dst_dt) return i;
    }
    return -1;
}

// One (transA, transB, b_signed) combination's worth of kernel storage:
// for each supported dst, an array of (n_cols -> unique_ptr<kernel>) for
// both the no-bias (nb) and with-bias (b) variants. The [kNumDstKinds]
// outer dimension and the [8] inner dimension (with [0] and [7] unused
// because n_cols ∈ [1, 6]) mirror the f32 GEMM dispatcher's shape so the
// patterns read consistently across kernels in this directory.
struct jit_rvv_gemm_s8_kernel_storage_t {
    std::array<std::array<std::unique_ptr<jit_rvv_gemm_s8_kernel_t>, 8>,
            kNumDstKinds>
            nb;
    std::array<std::array<std::unique_ptr<jit_rvv_gemm_s8_kernel_t>, 8>,
            kNumDstKinds>
            b;
};

template <bool isTransA, bool isTransB, bool b_signed>
const jit_rvv_gemm_s8_kernel_storage_t &get_jit_rvv_gemm_s8_kernel_storage() {
    static jit_rvv_gemm_s8_kernel_storage_t storage;
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        const auto &dts = supported_dst_dts();
        for (int dt = 0; dt < kNumDstKinds; ++dt) {
            const data_type_t dst_dt = dts[dt];
            for (dim_t n_cols = 1; n_cols <= 6; n_cols++) {
                storage.nb[dt][n_cols].reset(new jit_rvv_gemm_s8_kernel_t(
                        n_cols, isTransA, isTransB, b_signed, dst_dt, false));
                storage.b[dt][n_cols].reset(new jit_rvv_gemm_s8_kernel_t(
                        n_cols, isTransA, isTransB, b_signed, dst_dt, true));
            }
        }
    });
    return storage;
}

// Pick the per-(transA, transB, b_signed) storage. Exhaustive over the 8
// combinations; aborts on a combination the dispatcher macro doesn't
// cover rather than silently picking a wrong kernel.
const jit_rvv_gemm_s8_kernel_storage_t &pick_jit_rvv_gemm_s8_kernel_storage(
        bool isTransA, bool isTransB, bool b_signed) {
#define DISPATCH(SA, SB, BS) \
    if (isTransA == (SA) && isTransB == (SB) && b_signed == (BS)) { \
        return get_jit_rvv_gemm_s8_kernel_storage<SA, SB, BS>(); \
    }
    DISPATCH(false, false, true)
    DISPATCH(false, false, false)
    DISPATCH(false, true, true)
    DISPATCH(false, true, false)
    DISPATCH(true, false, true)
    DISPATCH(true, false, false)
    DISPATCH(true, true, true)
    DISPATCH(true, true, false)
#undef DISPATCH
    assert(!"unsupported (transA, transB, b_signed) combination");
    // The assert above only fires in debug builds. In release, abort
    // rather than silently picking a wrong kernel (the previous fallback
    // returned the s32 table, which would write s32 values for any
    // dst_dt — exactly the class of bug we want to surface).
    std::abort();
    // unreachable; satisfies the compiler's control-flow analysis.
    return get_jit_rvv_gemm_s8_kernel_storage<false, false, true>();
}

} // namespace

jit_rvv_gemm_s8_kernel_table_t get_jit_rvv_gemm_s8_kernel_table(
        bool isTransA, bool isTransB, bool b_signed, data_type_t dst_dt) {
    // Map the request to a dst index. Abort on an unsupported dst_dt
    // rather than silently substituting a different dst's kernel (the
    // previous code's s32 fallback had this exact hazard).
    const int idx = dst_kind_index(dst_dt);
    if (idx < 0) {
        assert(!"unsupported dst_dt for jit_rvv_gemm_s8 kernel table");
        std::abort();
    }

    const auto &storage
            = pick_jit_rvv_gemm_s8_kernel_storage(isTransA, isTransB, b_signed);

    // Build the per-n_cols lookup by reading pointers out of the
    // function-local-static storage. The `unique_ptr`s are populated once
    // under call_once and never reassigned, so the .get() pointers are
    // stable for the program's lifetime — copying them into the returned
    // table_t (12 raw-pointer copies) is correct.
    jit_rvv_gemm_s8_kernel_table_t table;
    for (dim_t n_cols = 1; n_cols <= 6; n_cols++) {
        table.nb[n_cols] = storage.nb[idx][n_cols].get();
        table.b[n_cols] = storage.b[idx][n_cols].get();
    }
    return table;
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
