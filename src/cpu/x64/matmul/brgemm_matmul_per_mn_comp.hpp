/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef CPU_X64_MATMUL_BRGEMM_MATMUL_PER_MN_COMP_HPP
#define CPU_X64_MATMUL_BRGEMM_MATMUL_PER_MN_COMP_HPP

#include <functional>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

// Zero-point compensation of the grouped int8 matmul.
//
// The int8 brgemm accumulates sum_k A'[m][k] * B'[k][n] of raw (uncentered)
// values in f32, where A' = A + 128 when an s8 src has to be shifted to u8
// (no native s8 x s8 and no unsigned B) and B' = B + int8_b_shift. Per-K
// scales sa (per row) and sb (per column), if applied in the kernel, scale
// every brgemm K call. Relative to the exact sum_k sa * sb * (A - za) * (B - zb)
// the accumulated sum carries, over every K range g where sa, sb, za and zb
// are constant, with T[m] = sum_k A[m][k], S[n] = sum_k B[k][n] and
// zb' = zb + int8_b_shift, the extra
//
//     sa * sb * ( zb'[n] * (T[m] - |g| * za[m]) + (za[m] + shift) * S[n] )
//
// Each term is rank-1 in (m, n), so the whole-K correction of a tile is one
// small f32 product that initializes the accumulation buffer:
//
//           K' = G wei rows [+ R src rows]
//       U (M x K')                            V (K' x N)
//       U[m][g]   = sum_{k in g} sa * (A[m][k] - za[m])
//                                             V[g][n]   = sb * zb'[n]
//       U[m][G+r] = sa * (za[m] + shift)      V[G+r][n] = sb * S_r[n]
//
//       buffer C (M_blk x N_blk, f32)  =  -U[m block] * V[.., n block]
//       buffer C                      +=  int8 brgemm, K call by K call
//
// The wei rows g partition K by the gcd of the wei scale (if in the kernel)
// and wei zp groups, the src rows r by the gcd of the wei scale, src scale
// and src zp groups; both are independent of the brgemm K blocking. U and V
// are prepared once per execution into the scratchpad (batch x M x K' and
// batch x K' x ldv floats; K' is a few tens of rows), the f32 brgemm runs
// once per output tile. The src rows need a pass over B (column sums by an
// AVX-512 kernel) and exist only with a src zero point or the src shift.
struct brgemm_matmul_per_mn_comp_t {
    // Fills the per_mn_comp_* fields of bgmmc. False when the compensation
    // cannot be computed this way; must run after the blocking is set up.
    static bool init_conf(brgemm_matmul_conf_t &bgmmc);
    static void book_scratchpad(memory_tracking::registrar_t &scratchpad,
            const brgemm_matmul_conf_t &bgmmc);
    // f32 brgemm descriptors for the M block and the M tail.
    static status_t init_brgemm_descs(
            brgemm_desc_t (&descs)[2], const brgemm_matmul_conf_t &bgmmc);
    status_t create_kernels(
            const brgemm_desc_t (&descs)[2], const brgemm_matmul_conf_t &bgmmc);

    // What prepare() reads from the matmul execution context.
    struct ctx_t {
        std::function<const char *(dim_t b)> A_batch, B_batch;
        std::function<const void *(dim_t b, dim_t k, dim_t m)> src_scales;
        std::function<const void *(dim_t b, dim_t k, dim_t m, int *nibble)>
                src_zp;
        std::function<const void *(dim_t b, dim_t k)> wei_scales;
        const void *wei_zp = nullptr;
        std::function<dim_t(dim_t b, dim_t k, dim_t n)> wei_zp_idx;
        float *U = nullptr;
        float *V = nullptr;
    };
    void prepare(const brgemm_matmul_conf_t &bgmmc, const ctx_t &ctx) const;

    static float *U_ptr(
            const brgemm_matmul_conf_t &bgmmc, float *U, dim_t b, dim_t m) {
        return U + (b * bgmmc.M + m) * bgmmc.per_mn_comp_K;
    }
    static float *V_ptr(
            const brgemm_matmul_conf_t &bgmmc, float *V, dim_t b, dim_t kk) {
        return V + (b * bgmmc.per_mn_comp_K + kk) * bgmmc.per_mn_comp_ldv;
    }

    // buffer C of the tile at rows [m, ..) x columns [n, n + n_size) = -U * V.
    void init_buffer_c(const brgemm_matmul_conf_t &bgmmc, const float *U_row,
            const float *V, int i_M, dim_t n, dim_t n_size, char *buf_C) const;

private:
    std::unique_ptr<brgemm_kernel_t> kernels_[2];
    std::unique_ptr<jit_generator_t> col_sum_kernel_;
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
