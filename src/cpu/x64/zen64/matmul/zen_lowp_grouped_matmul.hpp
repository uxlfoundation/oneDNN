/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_X64_ZEN64_MATMUL_ZEN_LOWP_GROUPED_MATMUL_HPP
#define CPU_X64_ZEN64_MATMUL_ZEN_LOWP_GROUPED_MATMUL_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

#if DNNL_X64_USE_ZEN
#include "lowoha_operators/matmul/lowoha_common.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace matmul {

#if DNNL_X64_USE_ZEN
namespace zen_lowp_gmm = zendnnl::lowoha::matmul;

// Per-argument (src/wei) static-quantization scale metadata resolved at
// primitive init() and consumed at execute() (the scale buffers live in the
// execution context). All experts share one scale buffer, so execute() slices
// it per expert using `elem_sz`; the per-expert dims are {M_g, ngroups_k} for
// the source and {ngroups_k, N} for the weights.
struct zen_lowp_grouped_scale_info_t {
    bool present = false;
    zendnnl::common::data_type_t dt = zendnnl::common::data_type_t::none;
    size_t elem_sz = 0;
    // Quantization groups along K: 1 for a per-token source or per-column
    // weight scale, K / group_size when the scale is blocked along K. Both
    // the per-expert slice offset and the dims handed to ZenDNN scale with it.
    dim_t ngroups_k = 1;
};
#endif

// Low-precision Zen grouped matmul (2Dx3D, MoE pattern): grouped src
// [total_M, K] x dense weights [G, K, N] -> grouped dst [total_M, N], run as
// one ZenDNN group_matmul_direct call with a descriptor per expert. Separate
// from the f32/bf16 zen_grouped_matmul_t so the floating-point path is
// unaffected.
// Symmetric static int8 only:
//   s8 src, s8 or s4 wei, f32/bf16 dst, s32 accumulation; no zero-points.
//   Scales: src and wei both required, both f32 or both bf16. Per-token src
//     with per-expert per-column or K-blocked wei, or K-blocked src with
//     K-blocked wei of the same group size. The wei K group (K when
//     per-column) is a multiple of 4 that divides K.
//   bias: f32/bf16, [G, N].
// Weights: plain abc/acb (abc only for s4), or zen_packed with the K group
// size of the weight scales; format_any resolves to zen_packed. Everything
// else declines to the reference grouped implementation.
struct zen_lowp_grouped_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T(
                "zen:grouped_matmul:lowp", zen_lowp_grouped_matmul_t);

        // Weight scales of [G, K, N] weights also vary along the expert
        // dimension, which the rank-based base helpers do not include.
        int wei_qmask_K() const { return (1 << 0) | (1 << 1); }
        int wei_qmask_N() const { return (1 << 0) | (1 << 2); }

        status_t init(const engine_t *engine);

    private:
        // Symmetric static int8 quantization validation: a matched pair of
        // source and weight scales, either ungrouped (per-token / per-expert
        // per-column) or blocked along K. Called from init().
        status_t init_int8_scales(const engine_t *engine);
    };

    zen_lowp_grouped_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    // Build the Zen eltwise post-op chain and resolve the scale metadata once
    // per primitive; both are expert-independent and reused for every group at
    // execute() time.
    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

#if DNNL_X64_USE_ZEN
    // Pre-built Zen post-op chain, copied into each group's params at
    // execute() time. Eltwise entries are complete; each binary_mul entry
    // carries only its static type, and its per-expert [M_g, N] operand
    // (buff / dims / leading_dim) is patched in at execute() time.
    std::vector<zen_lowp_gmm::matmul_post_op> postop_template_;
    // Per binary_mul post-op: slot in postop_template_, oneDNN post-op index
    // (to fetch the src1 buffer), src1 element size (to slice per expert),
    // whether src1 is grouped (its offsets are validated against dst at exec),
    // and the src1 element offset0 (applied to the allocation base).
    struct binary_src1_t {
        int chain_idx;
        int po_idx;
        size_t elem_sz;
        bool is_grouped;
        dim_t offset0;
    };
    std::vector<binary_src1_t> binary_src1_;

    // Static int8 scale metadata, sliced per expert at execute(): the source
    // scale spans [total_M, ngroups_k] and is sliced at the expert's row
    // offset, the weight scale spans [G, ngroups_k, N] and is sliced by expert.
    // pd_t::init() requires both, so `present` is always true for a
    // constructed primitive.
    zen_lowp_grouped_scale_info_t src_scale_;
    zen_lowp_grouped_scale_info_t wei_scale_;
#endif
};

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
