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

#ifndef CPU_RV64_GEMM_JIT_RVV_GEMM_S8_KERNEL_HPP
#define CPU_RV64_GEMM_JIT_RVV_GEMM_S8_KERNEL_HPP

#include "cpu/rv64/jit_generator.hpp"

#include <array>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

// RVV JIT micro-kernel for int8 GEMM on RV64.
//
// Computes an m x n_cols tile of:
//   C[0:m, 0:n_cols] = sum_k A[0:m, 0:K] * B[0:K, 0:n_cols]
//                    + (has_bias ? bias[0:m] : 0)
//
// dst_dt (data_type_t) selects the C element type and epilogue strategy.
// The driver (rvv_gemm_s8s8s32) enforces the alpha/beta contract for each
// destination before invoking the kernel, so this table only needs to
// document the kernel-side behavior:
//   - s32   : raw accumulator + optional s32 bias. The kernel applies
//             alpha and beta to the f32 acc only when beta!=0, so the
//             driver must reject anything other than alpha==1, beta in
//             {0,1}. Read-modify-write (beta==1) supports the K-split
//             partial-tile reduction path.
//   - f32   : full alpha/beta + f32 bias epilogue. Implemented entirely
//             in the JIT kernel; no driver-side alpha/beta restriction.
//   - s8    : s32 acc -> f32 -> +bias -> saturate to [-128, 127] -> RNE
//             round via vfcvt_x_f_v -> narrow to e8 -> vse8. Overwrite-
//             only: driver rejects alpha!=1 and beta!=0.
//   - u8    : same path as s8 but saturates to [0, 255] and uses
//             vfcvt_xu_f_v. Overwrite-only: driver rejects alpha!=1 and
//             beta!=0.
//   - f16   : s32 acc -> f32 -> alpha/beta/bias -> vfncvt_f_f_w -> vse16.
//             Requires Zvfh (gated by the driver via mayiuse(zvfh)).
//             Overwrite-only: driver rejects beta!=0 (the kernel's
//             read-modify-write path broadcasts C[0] to all M lanes via
//             vfmv.v.f, which is only correct when overwriting).
//   - bf16  : same path as f16 but uses vfncvtbf16_f_f_w. Requires
//             Zvfbfwma (gated by the driver via mayiuse(zvfbfwma)).
//             Overwrite-only: driver rejects beta!=0.
//
// Vector register layout (LMUL=m4 accumulator, same as before):
//   v0..v3    accumulator c0 (e32, LMUL=m4)
//   v4..v7    accumulator c1
//   v8..v11   accumulator c2
//   v12..v15  accumulator c3
//   v16..v19  accumulator c4
//   v20..v23  accumulator c5
//   v24..v25  A row buffer in e8 (LMUL=m1, v24) and e16 (LMUL=m2,
//              overlaps v24-v25)
//   v24..v27  scratch / C-load temporary (e32, LMUL=m4) — reuses the
//              v_a register group for the e32 epilogue (the K loop has
//              already exited by then).
//   v28..v31  bias loader (f32, e32 LMUL=m4)
struct jit_rvv_gemm_s8_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *A; // weights, s8 (M x K GEMM A axis)
        const void *B; // src, s8 or u8 (K x N GEMM B axis)
        void *C; // dst, see dst_dt above
        dim_t lda;
        dim_t ldb;
        dim_t ldc;
        dim_t K;
        dim_t m;
        float alpha; // applied for f32/f16/bf16 dst paths; ignored for
                //   s32 (must be 1.0) and s8/u8 (driver rejects !=1)
        float beta; // applied for f32 dst (any value); for f16/bf16 the
                //   driver rejects beta!=0; for s32 the driver requires
                //   beta in {0,1}; for s8/u8 the driver rejects beta!=0
        const float *bias; // optional, f32; broadcast per-row
        // True when the bias is a single scalar that broadcasts across the M
        // tile (bia_mask=0 / last dim == 1). The kernel then splats one float
        // instead of reading a full vector, which would overrun the
        // one-element object.
        int bias_is_scalar;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_gemm_s8_kernel_t)

    jit_rvv_gemm_s8_kernel_t(dim_t n_cols, bool isTransA, bool isTransB,
            bool b_signed, data_type_t dst_dt, bool has_bias);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    dim_t n_cols_;
    bool isTransA_;
    bool isTransB_;
    bool b_signed_; // true: B is s8; false: B is u8
    // dst_dt selects the C element type and epilogue strategy. Mirrors the
    // public data_type_t enum (see common/c_types_map.hpp) so the JIT kernel
    // and the driver (rvv_gemm_s8s8s32) can compare against the same constants
    // — no parallel integer encoding to drift out of sync.
    data_type_t dst_dt_;
    bool has_bias_;
};

// Per-n_cols lookup table of (with/without-bias) kernel pointers. The
// kernel pointers reference kernels owned by the per-(transA, transB,
// b_signed) function-local-static storage inside jit_rvv_gemm_s8_kernel.cpp;
// they remain valid for the program's lifetime (the storage is built once
// under std::call_once and never destroyed).
struct jit_rvv_gemm_s8_kernel_table_t {
    std::array<const jit_rvv_gemm_s8_kernel_t *, 8> nb {};
    std::array<const jit_rvv_gemm_s8_kernel_t *, 8> b {};
};

// Returns the per-n_cols lookup table of JIT kernels specialized for the
// given (transA, transB, B-signedness, dst_dt) combination. dst_dt must be
// one of {s32, f32, s8, u8, f16, bf16}; rvv_gemm_s8s8s32() guards that set
// before calling. The result is built on the fly (twelve pointer copies)
// from the owned kernels, so callers should cache the returned value at
// the dispatch site rather than calling per-tile.
jit_rvv_gemm_s8_kernel_table_t get_jit_rvv_gemm_s8_kernel_table(
        bool isTransA, bool isTransB, bool b_signed, data_type_t dst_dt);

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
