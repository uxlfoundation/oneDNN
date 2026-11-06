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

#ifndef CPU_X64_SDPA_SDPA_FULL_SOFTMAX_HPP
#define CPU_X64_SDPA_SDPA_FULL_SOFTMAX_HPP

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/optional.hpp"
#include "common/sdpa_pd.hpp"

#include "cpu/x64/brgemm/brgemm_containers.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;

namespace cpu {
namespace x64 {

class jit_generator_t;
namespace sdpa_brgemm_copy {
class b_packer_t;
} // namespace sdpa_brgemm_copy
namespace softmax_impl {
struct jit_softmax_kernel_base_t;
} // namespace softmax_impl

// -----------------------------------------------------------------------------
// SDPA with softmax over the complete key axis (x64):
//   scores = Q*K^T ; P = softmax(scale*scores [+ select-mask], over seq_kv)
//   out = P*V
// Block over seq_q into q_block tiles. Each tile computes scores for all seq_kv
// keys, then applies softmax over that axis using the JIT softmax kernel.
// Parallelize over batch x query heads (one head or one KV-head group per work
// item, see sdpa_full_softmax_schedule_t) x query-tiles.
// -----------------------------------------------------------------------------

// One mm1 (QK^T) post-op in chain order, folded into the BRGEMM store: binary
// entries apply their algorithm to an RHS; eltwise entries apply an activation.
struct sdpa_mm1_post_op_t {
    dnnl::impl::alg_kind_t alg = dnnl::impl::alg_kind::undef;
    bool is_binary = false; // false => eltwise
    // Eltwise parameters (is_binary == false).
    float alpha = 0.0f;
    float beta = 0.0f;
    // Binary rhs: a scalar uses a [1 x 1] descriptor and its pointer is not
    // query-tile-offset. A tensor rhs uses a query-row by key-position tile;
    // dimensions and strides are derived from its PD-owned memory descriptor.
    bool rhs_is_scalar = true;
    dnnl::impl::data_type_t rhs_dt = dnnl::impl::data_type::f32;
    std::vector<dim_t> rhs_dims, rhs_strides;
};

// Work decomposition. per_head: a work item is one query block of one query
// head. kv_head_group: a work item is one query block of all query heads that
// share a KV head, so each K/V tile is read once for the whole group.
enum class sdpa_full_softmax_schedule_t { any, per_head, kv_head_group };

// Where K and V are packed when BRGEMM needs them packed. ahead: whole tensors
// are packed into shared scratch before the main loop. per_tile: a work item
// packs each tile into its own scratch right before using it.
enum class sdpa_full_softmax_kv_packing_t { any, ahead, per_tile };

struct sdpa_full_softmax_params_t {
    dim_t batch = 0;
    dim_t num_head_q = 0;
    // GQA group size: num_head_q / num_head_kv (>= 1).
    dim_t group_head = 1;
    dim_t seq_q = 0;
    dim_t seq_kv = 0;
    dim_t hs_qk = 0;
    dim_t hs_v = 0;

    // Compute data type for Q/K/V and the mm1/mm2 BRGEMM inputs (f32/f16/bf16).
    // The scores and pv tiles are always accumulated in f32 (BRGEMM C is f32);
    // the transposed-K buffer and P (mm2's A operand) are materialised in this
    // type, and the softmax runs on the f32 scores.
    dnnl::impl::data_type_t mm_dt = dnnl::impl::data_type::f32;
    // SDPA output (result) data type; the f32 pv tile is down-converted to it.
    dnnl::impl::data_type_t out_dt = dnnl::impl::data_type::f32;

    // Derived source layout metadata used while configuring BRGEMM and packing.
    std::vector<dim_t> q_strides, k_strides, v_strides, o_strides;
    std::vector<dim_t> cond_strides, cond_dims;

    bool has_select = false;
    // When true, keep scores where cond == 0; otherwise keep where cond != 0.
    bool invert_select = false;

    // SoftMax "inf_as_zero" mode: a fully-masked row (all -inf) produces an
    // all-zero probability row instead of NaN. Selects the softmax primitive
    // alg_kind (softmax_accurate_inf_as_zero vs softmax_accurate).
    bool softmax_inf_as_zero = false;

    // Supported mm1 (QK^T) post-ops, applied in order: scale then attention
    // mask. Select is handled separately. These ops are sliced to the query-tile
    // shape and folded into the BRGEMM store. Soft-cap is not wired into this path.
    // Empty when mm1 has no post-ops.
    std::vector<sdpa_mm1_post_op_t> mm1_post_ops;

    // Requested strategies; `any` lets configure() choose.
    sdpa_full_softmax_schedule_t schedule = sdpa_full_softmax_schedule_t::any;
    sdpa_full_softmax_kv_packing_t kv_packing
            = sdpa_full_softmax_kv_packing_t::any;
};

template <typename T>
struct sdpa_arg_buf_t {
    T *buffer = nullptr;
    const memory_desc_t *md = nullptr;
};

// Runtime pointers / scalars, resolved per execute call.
struct sdpa_full_softmax_run_args_t {
    sdpa_arg_buf_t<const char> q, k, v, cond;
    sdpa_arg_buf_t<char> out;

    void *scratch = nullptr;
    float fill = 0.0f;
    // Base pointers for the mm1 binary post-op right-hand sides, one per binary
    // entry in sdpa_full_softmax_params_t::mm1_post_ops (in the same order; eltwise
    // entries consume none). Each is offset per (batch, head, query-tile) using
    // the entry's dims/strides. For a scalar rhs (e.g. the QK scale) the pointer
    // addresses a single element and is not offset.
    std::vector<const void *> mm1_post_op_rhs;
};

struct tensor_copy_desc_t {
    size_t scratch_offset = 0;
    size_t batch_stride = 0;
    size_t head_stride = 0;
    dim_t padded_k = 0;
    dim_t vnni_factor = 1;
};

// Layout of the tile buffers in one thread's scratch block. The score tiles
// start at offset 0; unused areas have zero size.
struct sdpa_full_softmax_thread_scratch_t {
    bool has_prob = false;
    size_t pv = 0, prob = 0, amx_wsp = 0;
    size_t size = 0;

    void set_compute_type(data_type_t dt, size_t dt_sz) {
        has_prob = dt != data_type::f32;
        prob_dt_size_ = dt_sz;
    }

    size_t score_tile_row_bytes(dim_t ld_scores, dim_t ld_prob) const {
        return static_cast<size_t>(ld_scores) * sizeof(float)
                + (has_prob ? static_cast<size_t>(ld_prob) * prob_dt_size_ : 0);
    }

    void set_tile_buffers(size_t item_q_rows, dim_t ld_scores, dim_t ld_prob,
            dim_t ld_pv, bool mm2_direct, size_t amx_wsp_bytes) {
        const size_t scores_bytes
                = align64(item_q_rows * ld_scores * sizeof(float));
        const size_t pv_bytes
                = mm2_direct ? 0 : align64(item_q_rows * ld_pv * sizeof(float));
        const size_t prob_bytes
                = has_prob ? align64(item_q_rows * ld_prob * prob_dt_size_) : 0;
        pv = scores_bytes;
        prob = pv + pv_bytes;
        amx_wsp = prob + prob_bytes;
        size = amx_wsp + amx_wsp_bytes;
    }

private:
    static size_t align64(size_t n) {
        return (n + 63) & ~static_cast<size_t>(63);
    }

    size_t prob_dt_size_ = 0;
};

struct sdpa_full_softmax_brgemm_t {
    std::shared_ptr<brgemm_containers::brgemm_desc_container_t> descs;
    // Canonical descriptor indices; -1 denotes an unused variant.
    // mm1: [is_q_tail][is_n_tail], mm2: [is_q_tail][is_k_tail][is_v_tail].
    int mm1[2][2] = {{-1, -1}, {-1, -1}};
    int mm2[2][2][2] = {{{-1, -1}, {-1, -1}}, {{-1, -1}, {-1, -1}}};
    // Beta-zero mm2 for the first full K block when the reduction is tiled.
    int mm2_beta0[2][2] = {{-1, -1}, {-1, -1}};
};

// Derived compute configuration: the JIT-free arithmetic (query/KV blocking,
// per-thread + global scratch layout). Computed once by configure() and owned
// by the primitive_desc, which also sizes its scratchpad from this config.
struct sdpa_full_softmax_conf_t {
    sdpa_full_softmax_params_t params;
    // Resolved strategies (never `any`).
    sdpa_full_softmax_schedule_t schedule
            = sdpa_full_softmax_schedule_t::per_head;
    sdpa_full_softmax_kv_packing_t kv_packing
            = sdpa_full_softmax_kv_packing_t::ahead;
    // Query heads per work item: 1, or group_head for kv_head_group.
    dim_t heads_per_item = 1;
    dim_t q_block = 0, q_tail = 0;
    // mm1 writes the score tile in mm1_n_block-wide column blocks,
    // mm2 reduces P in mm2_k_block-deep blocks.
    dim_t mm1_n_block = 0, mm1_n_tail = 0;
    dim_t mm2_k_block = 0, mm2_k_tail = 0;
    // mm2 accumulates across its seq_kv blocks: beta = 0 for a single block,
    // else the first block uses a beta = 0 kernel and the rest beta = 1.
    float mm2_beta = 0.0f;
    // mm2 writes its f32 result directly into the user output (no pv+scatter).
    bool mm2_direct = false;
    bool mm1_needs_vnni_b = false;
    bool mm2_needs_vnni_b = false;
    // K / V are packed for BRGEMM; kv_packing tells where.
    bool pack_k = false, pack_v = false;
    dim_t ldk = 0, ldv = 0;
    dim_t ld_scores = 0, ld_prob = 0, ld_pv = 0;
    dim_t v_block = 0, v_tail = 0;
    bool pack_k_for_aliasing = false;
    // Fold non-inverted select into mm1 for dense condition rows or a single
    // query row.
    bool mm1_select_postop = false;
    size_t amx_wsp_bytes = 0;
    sdpa_full_softmax_thread_scratch_t thread_scratch;
    size_t scratch_per_thread = 0;
    size_t copy_buffer_bytes = 0;
    int nthr = 0;

    sdpa_full_softmax_brgemm_t brgemm;

    // The optional copy descriptors describe whole tensors packed ahead into
    // shared scratch.
    utils::optional_t<tensor_copy_desc_t> q_copy_desc, k_copy_desc, v_copy_desc,
            o_copy_desc, cond_copy_desc;
    // Tile buffers for packing per tile: scratch_offset is relative to the
    // thread's scratch block, padded_k is that of a full (non-tail) tile.
    utils::optional_t<tensor_copy_desc_t> k_tile_desc, v_tile_desc;

    size_t scratch_total(int n) const {
        return scratch_per_thread * static_cast<size_t>(n) + copy_buffer_bytes;
    }
};

// Compiled kernels and AMX palettes for the full-softmax SDPA, owned by the
// primitive.
struct sdpa_full_softmax_kernels_t {
    // The packer type is forward-declared, so construction/destruction is
    // defined where it is complete.
    sdpa_full_softmax_kernels_t();
    ~sdpa_full_softmax_kernels_t();
    sdpa_full_softmax_kernels_t(const sdpa_full_softmax_kernels_t &) = delete;
    sdpa_full_softmax_kernels_t &operator=(const sdpa_full_softmax_kernels_t &)
            = delete;

    brgemm_containers::brgemm_kernel_container_t brgemm_kernels;
    brgemm_containers::brgemm_palette_container_t brgemm_palettes;
    // K / V packers. Packing ahead uses [0] for the whole tensor; packing per
    // tile uses [is_tail] for one tile (the copy kernels bake in their shape).
    std::unique_ptr<sdpa_brgemm_copy::b_packer_t> k_packers[2], v_packers[2];

    // Vectorized JIT softmax kernel and its pd. execute() invokes the kernel
    // for each score row.
    std::shared_ptr<primitive_desc_t> softmax_pd;
    std::shared_ptr<softmax_impl::jit_softmax_kernel_base_t> softmax_kernel;

    // Standalone JIT select-mask pre-pass kernel (null unless a select mask is
    // present and not folded into mm1); the condition is dense along seq_kv
    // (cond_col == 1, guaranteed by the pd).
    std::shared_ptr<jit_generator_t> select_kernel;
};

// Query-blocked SDPA with softmax over the complete key axis, folded into the
// CPU primitive: the primitive_desc owns the configuration and descriptors,
// while the primitive owns the kernels and drives execution.
namespace sdpa_full_softmax {

// Compute the derived configuration (query/KV blocking, per-thread + global
// scratch layout) and build the finalized BRGEMM descriptors. JIT-free, so a
// primitive_desc can size its scratchpad without compiling kernels.
status_t configure(const sdpa_full_softmax_params_t &params,
        sdpa_full_softmax_conf_t &conf);

// JIT-compile the BRGEMM + softmax kernels from the configuration produced by
// configure(), into `kernels`. The engine instantiates the reused jit softmax
// kernel; the execute path stays engine-free.
status_t create_kernels(cpu_isa_t isa_for_ir,
        const sdpa_full_softmax_conf_t &conf, engine_t *engine,
        sdpa_full_softmax_kernels_t &kernels);

status_t execute(const sdpa_full_softmax_conf_t &conf,
        const sdpa_full_softmax_kernels_t &kernels,
        const sdpa_full_softmax_run_args_t &args, int nthr);

} // namespace sdpa_full_softmax

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
