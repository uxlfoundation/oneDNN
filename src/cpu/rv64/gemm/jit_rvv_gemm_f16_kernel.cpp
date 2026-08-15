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

#include "cpu/rv64/gemm/jit_rvv_gemm_f16_kernel.hpp"

#include <memory>
#include <mutex>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

using namespace Xbyak_riscv;
using namespace dnnl::impl::utils;

jit_rvv_gemm_f16_kernel_t::jit_rvv_gemm_f16_kernel_t(
        dim_t n_cols, bool isTransA, data_type_t in_dt)
    : jit_generator_t("rv64_gemm_kernel_f16_jit")
    , n_cols_(n_cols)
    , isTransA_(isTransA)
    , is_bf16_(in_dt == data_type::bf16) {
    assert(utils::one_of(in_dt, data_type::f16, data_type::bf16));
    create_kernel();
}

void jit_rvv_gemm_f16_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;

    const Reg reg_A_ptr = a1; // running pointer into A
    const Reg reg_m = a2; // tile height (used for vsetvli)
    const Reg reg_C_base = a3; // base pointer to C(:, 0)

    const Reg reg_lda_bytes = t0;
    const Reg reg_ldb_bytes = t1;
    const Reg reg_ldc_bytes = t2;
    const Reg reg_K = t3;

    const Reg reg_k = a4; // current k counter
    const Reg reg_B0_ptr = a6; // running pointer into B
    const Reg reg_tmp0 = a7;
    const FReg freg_b[6] = {fa2, fa3, fa4, fa5, fa6, fa7};

    const VReg v_c[6]
            = {VReg(0), VReg(4), VReg(8), VReg(12), VReg(16), VReg(20)};
    const VReg v_a0(24); // e16/m2: v24-v25
    const VReg v_a1(28); // e16/m2: v28-v29

    // Layout of call_params_t:
    //   0  : const void *A
    //   8  : const void *B
    //   16 : void *C
    //   24 : dim_t lda
    //   32 : dim_t ldb
    //   40 : dim_t ldc
    //   48 : dim_t K
    //   56 : dim_t m
    ld(reg_A_ptr, reg_param, 0);
    ld(reg_B0_ptr, reg_param, 8);
    ld(reg_C_base, reg_param, 16);
    ld(reg_lda_bytes, reg_param, 24);
    ld(reg_ldb_bytes, reg_param, 32);
    ld(reg_ldc_bytes, reg_param, 40);
    ld(reg_K, reg_param, 48);
    ld(reg_m, reg_param, 56);

    // A, B and C elements are 2 bytes.
    slli(reg_lda_bytes, reg_lda_bytes, 1);
    slli(reg_ldb_bytes, reg_ldb_bytes, 1);
    slli(reg_ldc_bytes, reg_ldc_bytes, 1);

    const Reg &reg_tmp3 = reg_param;

    // f32 accumulators: e32/m4. Zero the full groups before switching to the
    // e16/m2 configuration of the K loop.
    vsetvli(x0, reg_m, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    for (dim_t c = 0; c < n_cols_; c++)
        vmv_v_i(v_c[c], 0);

    // A rows and the widening FMA run at e16/m2 (same VLMAX as e32/m4).
    vsetvli(x0, reg_m, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);

    auto emit_load_a = [&](const VReg &v_a) {
        if (isTransA_) {
            vlse16_v(v_a, reg_A_ptr, reg_lda_bytes);
        } else {
            vle16_v(v_a, reg_A_ptr);
        }
    };

    auto emit_advance_a = [&]() {
        if (isTransA_) {
            addi(reg_A_ptr, reg_A_ptr, 2);
        } else {
            add(reg_A_ptr, reg_A_ptr, reg_lda_bytes);
        }
    };

    auto emit_advance_b = [&]() { addi(reg_B0_ptr, reg_B0_ptr, 2); };

    auto emit_fma = [&](dim_t col, const VReg &v_a) {
        if (is_bf16_) {
            vfwmaccbf16_vf(v_c[col], freg_b[col], v_a);
        } else {
            vfwmacc_vf(v_c[col], freg_b[col], v_a);
        }
    };

    // Load the n_cols B scalars (stride ldb apart), interleaving each flh with
    // the FMA of the previously loaded scalar to hide the load latency.
    auto emit_load_b_scattered_compute = [&](const VReg &v_a) {
        flh(freg_b[0], reg_B0_ptr, 0);
        if (n_cols_ == 1) {
            emit_fma(0, v_a);
        } else {
            add(reg_tmp0, reg_B0_ptr, reg_ldb_bytes);
            flh(freg_b[1], reg_tmp0, 0);
            emit_fma(0, v_a);
            for (dim_t c = 2; c < n_cols_; c++) {
                add(reg_tmp0, reg_tmp0, reg_ldb_bytes);
                flh(freg_b[c], reg_tmp0, 0);
                emit_fma(c - 1, v_a);
            }
            emit_fma(n_cols_ - 1, v_a);
        }
    };

    mv(reg_k, x0);

    Label label_k_done;
    Label label_loop_a0, label_loop_a1;
    Label label_drain_a0, label_drain_a1;

    bge(reg_k, reg_K, label_k_done);

    emit_load_a(v_a0);
    addi(reg_k, reg_k, 1);
    bge(reg_k, reg_K, label_drain_a0);

    L(label_loop_a0);
    emit_advance_a();
    emit_load_a(v_a1);
    emit_load_b_scattered_compute(v_a0);
    emit_advance_b();
    addi(reg_k, reg_k, 1);
    bge(reg_k, reg_K, label_drain_a1);

    L(label_loop_a1);
    emit_advance_a();
    emit_load_a(v_a0);
    emit_load_b_scattered_compute(v_a1);
    emit_advance_b();
    addi(reg_k, reg_k, 1);
    bge(reg_k, reg_K, label_drain_a0);
    j_(label_loop_a0);

    L(label_drain_a0);
    emit_load_b_scattered_compute(v_a0);
    j_(label_k_done);

    L(label_drain_a1);
    emit_load_b_scattered_compute(v_a1);
    j_(label_k_done);

    L(label_k_done);

    // C store: narrow the f32 accumulators in place to the input data type and
    // store (overwrite-only; the driver rejects beta != 0). Stays at e16/m2,
    // the configuration of the narrowing destination.
    for (dim_t c = 0; c < n_cols_; c++) {
        if (c == 0) {
            mv(reg_tmp3, reg_C_base);
        } else {
            li(reg_tmp0, c);
            mul(reg_tmp3, reg_ldc_bytes, reg_tmp0);
            add(reg_tmp3, reg_C_base, reg_tmp3);
        }
        if (is_bf16_) {
            vfncvtbf16_f_f_w(v_c[c], v_c[c]);
        } else {
            vfncvt_f_f_w(v_c[c], v_c[c]);
        }
        vse16_v(v_c[c], reg_tmp3);
    }

    ret();
#else
    ret();
#endif
}

namespace {

template <bool isTransA, bool isBf16>
struct jit_rvv_gemm_f16_kernel_storage_t {
    std::array<std::unique_ptr<jit_rvv_gemm_f16_kernel_t>, 8> nb;
    jit_rvv_gemm_f16_kernel_table_t table;
};

template <bool isTransA, bool isBf16>
jit_rvv_gemm_f16_kernel_storage_t<isTransA, isBf16> &
get_jit_rvv_gemm_f16_kernel_storage() {
    static jit_rvv_gemm_f16_kernel_storage_t<isTransA, isBf16> storage;
    static std::once_flag initialized;

    std::call_once(initialized, [] {
        for (dim_t n_cols = 1; n_cols <= 6; n_cols++) {
            storage.nb[n_cols].reset(new jit_rvv_gemm_f16_kernel_t(n_cols,
                    isTransA, isBf16 ? data_type::bf16 : data_type::f16));
            storage.table.nb[n_cols] = storage.nb[n_cols].get();
        }
    });

    return storage;
}

} // namespace

const jit_rvv_gemm_f16_kernel_table_t &get_jit_rvv_gemm_f16_kernel_table(
        bool isTransA, data_type_t in_dt) {
    if (isTransA) {
        return in_dt == data_type::bf16
                ? get_jit_rvv_gemm_f16_kernel_storage<true, true>().table
                : get_jit_rvv_gemm_f16_kernel_storage<true, false>().table;
    }
    return in_dt == data_type::bf16
            ? get_jit_rvv_gemm_f16_kernel_storage<false, true>().table
            : get_jit_rvv_gemm_f16_kernel_storage<false, false>().table;
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
