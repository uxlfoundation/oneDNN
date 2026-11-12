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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "common/bfloat16.hpp"
#include "common/dnnl_thread.hpp"
#include "common/float16.hpp"
#include "common/memory_desc.hpp"
#include "common/opdesc.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/x64/sdpa/sdpa_full_softmax.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_uni_softmax.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"
#include "cpu/x64/sdpa/sdpa_brgemm_copy.hpp"
#include "cpu/x64/sdpa/sdpa_full_softmax_select_ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

inline size_t align64(size_t n) {
    return (n + 63) & ~static_cast<size_t>(63);
}

// Return the physical row stride in elements. Add one cache line when the
// row stride would otherwise alias L1 sets; keep short rows dense.
dim_t scratch_row_ld(dim_t width, size_t dt_sz, bool allow_padding) {
    const size_t line = platform::get_cache_line_size();
    const size_t l1_way_bytes = platform::get_num_sets_in_cache(1) * line;
    const size_t row_bytes = static_cast<size_t>(width) * dt_sz;
    if (!allow_padding || l1_way_bytes == 0 || row_bytes % l1_way_bytes != 0
            || row_bytes < 2 * l1_way_bytes)
        return width;
    return static_cast<dim_t>((row_bytes + line) / dt_sz);
}

// Returns the peak load of an f32 tile on the L2 cache: the largest number of
// tile lines mapped to one L2 set, as a fraction of the L2 associativity
// (1.0 means the tile alone fills a whole set). Returns 0 if unknown.
float f32_tile_l2_set_load(
        dim_t tile_rows, dim_t tile_columns, dim_t row_stride) {
    const size_t sets = platform::get_num_sets_in_cache(2);
    const size_t ways = platform::get_num_ways_in_cache(2);
    const size_t line_size = platform::get_cache_line_size();
    if (!sets || !ways || !line_size || tile_rows <= 0 || tile_columns <= 0)
        return 0.f;

    const size_t period = sets * line_size;
    const size_t stride = (row_stride * sizeof(float)) % period;
    const size_t tile_width = tile_columns * sizeof(float);
    std::vector<size_t> occupancy(sets);
    size_t max_occupancy = 0;
    size_t offset = 0;
    for (dim_t row = 0; row < tile_rows; ++row) {
        const size_t first_line = offset / line_size;
        const size_t last_line = (offset + tile_width - 1) / line_size;
        for (size_t line = first_line; line <= last_line; ++line)
            max_occupancy = nstl::max(max_occupancy, ++occupancy[line % sets]);
        offset = (offset + stride) % period;
    }
    return static_cast<float>(max_occupancy) / ways;
}

// Tensor descriptor for a buffer, its memory descriptor, optional copy
// descriptor, and scratch pointer. Templated to distinguish between const
// and non-const byte buffers.
template <typename T>
struct tensor_desc_t {
    sdpa_arg_buf_t<T> arg;
    const tensor_copy_desc_t *copy_desc = nullptr;
    char *scratch_ptr = nullptr;
};

template <typename T>
tensor_desc_t<T> get_tensor_desc(const sdpa_arg_buf_t<T> &arg,
        const utils::optional_t<tensor_copy_desc_t> &copy_desc,
        char *scratch_ptr, size_t shared_area_offset) {
    tensor_desc_t<T> td;
    td.arg = arg;
    if (copy_desc) {
        td.copy_desc = &*copy_desc;
        td.scratch_ptr
                = scratch_ptr + shared_area_offset + copy_desc->scratch_offset;
    }
    return td;
}

// Get the pointer to the argument buffer for the given batch and head indices.
template <typename T>
T *get_arg_ptr(const tensor_desc_t<T> &td, dim_t batch_idx, dim_t head_idx,
        dim_t row_idx = 0) {
    const memory_desc_wrapper mdw(td.arg.md);
    dim_t offset = 0;
    if (mdw.dims()[0] != 1) offset += batch_idx * mdw.strides()[0];
    if (mdw.dims()[1] != 1) offset += head_idx * mdw.strides()[1];
    if (mdw.dims()[2] != 1) offset += row_idx * mdw.strides()[2];
    return td.arg.buffer + offset * types::data_type_size(mdw.data_type());
}

// Get the pointer to the copy buffer for the given batch and head indices.
template <typename T>
char *get_copy_ptr(
        const tensor_desc_t<T> &td, dim_t batch_idx, dim_t head_idx) {
    const auto *copy_desc = td.copy_desc;
    assert(copy_desc);
    return td.scratch_ptr + batch_idx * copy_desc->batch_stride
            + head_idx * copy_desc->head_stride;
}

// Get the pointer to the tile buffer for the given batch and head indices
// depending on whether a copy buffer is used.
template <typename T>
T *get_tile_ptr(const tensor_desc_t<T> &td, dim_t batch_idx, dim_t head_idx) {
    if (td.copy_desc) return get_copy_ptr(td, batch_idx, head_idx);
    return get_arg_ptr(td, batch_idx, head_idx);
}

// Return the B-matrix offset in elements for logical coordinates [k0, n0].
template <typename T>
size_t get_tile_offset_in_elements(
        const tensor_desc_t<T> &td, dim_t k0, dim_t n0) {
    if (!td.copy_desc) {
        constexpr int row_dim = sdpa_pd_t::ndims - 2;
        constexpr int col_dim = sdpa_pd_t::ndims - 1;
        const memory_desc_wrapper mdw(td.arg.md);
        return static_cast<size_t>(k0) * mdw.strides()[row_dim]
                + static_cast<size_t>(n0) * mdw.strides()[col_dim];
    }

    return sdpa_brgemm_copy::b_packer_t::panel_offset(
            k0, n0, td.copy_desc->padded_k, td.copy_desc->vnni_factor);
}

// B operand (K for mm1, V for mm2) of one KV head, addressed by logical
// [k, n]. It is read in place or from the tensor packed ahead. When packing
// per tile, load_tile() packs one tile into the thread's buffer, and at() may
// address only that tile.
class b_operand_t {
public:
    using packer_t = sdpa_brgemm_copy::b_packer_t;

    b_operand_t(const tensor_desc_t<const char> &td, dim_t batch_idx,
            dim_t kv_head_idx, const std::unique_ptr<packer_t> (&packers)[2],
            const utils::optional_t<tensor_copy_desc_t> &tile_desc,
            char *thread_scratch, size_t dt_sz)
        : td_(td)
        , head_(get_tile_ptr(td, batch_idx, kv_head_idx))
        , packers_(packers)
        , tile_desc_(tile_desc ? &*tile_desc : nullptr)
        , tile_(tile_desc ? thread_scratch + tile_desc->scratch_offset
                          : nullptr)
        , dt_sz_(dt_sz) {}

    // Packs the tile starting at [k0, n0], a full block or the tail block, when
    // packing per tile; no-op otherwise.
    void load_tile(dim_t k0, dim_t n0, bool is_tail) {
        if (!tile_desc_) return;
        const packer_t &packer = *packers_[is_tail];
        packer.execute(
                head_ + get_tile_offset_in_elements(td_, k0, n0) * dt_sz_,
                tile_);
        tile_k0_ = k0;
        tile_n0_ = n0;
        tile_padded_k_ = packer.padded_k();
    }

    const char *at(dim_t k, dim_t n) const {
        if (!tile_desc_)
            return head_ + get_tile_offset_in_elements(td_, k, n) * dt_sz_;
        assert(k >= tile_k0_ && n >= tile_n0_);
        return tile_
                + packer_t::panel_offset(k - tile_k0_, n - tile_n0_,
                          tile_padded_k_, tile_desc_->vnni_factor)
                * dt_sz_;
    }

private:
    const tensor_desc_t<const char> &td_;
    const char *head_;
    const std::unique_ptr<packer_t> (&packers_)[2];
    const tensor_copy_desc_t *tile_desc_;
    char *tile_;
    size_t dt_sz_;
    dim_t tile_k0_ = 0, tile_n0_ = 0, tile_padded_k_ = 0;
};

// Calls f(start, is_tail) for each block of [begin, end); only the last block
// can be shorter (a tail).
template <typename F>
void for_each_block(dim_t begin, dim_t end, dim_t block, const F &f) {
    for (dim_t start = begin; start < end; start += block)
        f(start, end - start < block);
}

// Appends the mm1 post-op right-hand sides of one query head in chain order:
// one per binary in mm1_post_ops, then the fill scalar and condition of a
// folded select. A scalar rhs is used as is; a tensor rhs is offset by batch,
// head and the first query row q0. Its key column comes from the output
// pointer at execution.
void append_mm1_rhs(std::vector<const void *> &rhs,
        const sdpa_full_softmax_params_t &p,
        const sdpa_full_softmax_run_args_t &args, dim_t batch_idx,
        dim_t query_head_idx, dim_t q0, bool select_in_mm1,
        const uint8_t *cond) {
    for (size_t pi = 0; pi < p.mm1_post_ops.size(); ++pi) {
        const auto &pop = p.mm1_post_ops[pi];
        if (!pop.is_binary) continue;
        const char *base = static_cast<const char *>(args.mm1_post_op_rhs[pi]);
        if (!pop.rhs_is_scalar) {
            // The RHS is [batch, query_head, query, key]; broadcast axes
            // (dim == 1) contribute nothing.
            const auto &d = pop.rhs_dims;
            const auto &s = pop.rhs_strides;
            dim_t off = 0;
            if (d[0] != 1) off += batch_idx * s[0];
            if (d[1] != 1) off += query_head_idx * s[1];
            if (d[sdpa_pd_t::mask_q_index] != 1)
                off += q0 * s[sdpa_pd_t::mask_q_index];
            base += off * types::data_type_size(pop.rhs_dt);
        }
        rhs.push_back(base);
    }
    if (select_in_mm1) {
        rhs.push_back(&args.fill);
        rhs.push_back(cond);
    }
}

// Writes an f32 [m x n] tile to the user output, converting to out_dt.
// Unit-stride rows use the bulk converters; a strided output column falls
// back to scalar stores.
void store_pv_tile(const float *pv, dim_t ld_pv, char *out, dim_t o_row_stride,
        dim_t o_col_stride, data_type_t out_dt, dim_t m, dim_t n) {
    const size_t o_dt_sz = types::data_type_size(out_dt);
    const bool dense_col = o_col_stride == 1;
    for (dim_t i = 0; i < m; ++i) {
        const float *prow = pv + i * ld_pv;
        char *out_row = out + i * o_row_stride * o_dt_sz;
        if (out_dt == data_type::f32) {
            float *dst = reinterpret_cast<float *>(out_row);
            if (dense_col)
                std::memcpy(dst, prow, static_cast<size_t>(n) * sizeof(float));
            else
                for (dim_t d = 0; d < n; ++d)
                    dst[d * o_col_stride] = prow[d];
        } else if (out_dt == data_type::f16) {
            float16_t *dst = reinterpret_cast<float16_t *>(out_row);
            if (dense_col)
                cvt_float_to_float16(dst, prow, static_cast<size_t>(n));
            else
                for (dim_t d = 0; d < n; ++d)
                    dst[d * o_col_stride] = float16_t(prow[d]);
        } else if (out_dt == data_type::bf16) {
            bfloat16_t *dst = reinterpret_cast<bfloat16_t *>(out_row);
            if (dense_col)
                cvt_float_to_bfloat16(dst, prow, static_cast<size_t>(n));
            else
                for (dim_t d = 0; d < n; ++d)
                    dst[d * o_col_stride] = bfloat16_t(prow[d]);
        }
    }
}

// Build a finalized BRGEMM descriptor.
status_t build_brgemm_desc(brgemm_desc_t &brg, data_type_t dt, float beta,
        dim_t M, dim_t N, dim_t K, dim_t lda, dim_t ldb, dim_t ldc,
        const std::vector<sdpa_mm1_post_op_t> *post_ops = nullptr,
        bool select_as_postop = false, dim_t full_output_width = 0,
        bool b_is_vnni = false, bool stack_heads = false) {
    CHECK(brgemm_desc_init(&brg, isa_undef, brgemm_addr, dt, dt,
            /*transA=*/false, /*transB=*/false, brgemm_row_major,
            /*alpha=*/1.0f, beta, lda, ldb, ldc, M, N, K, /*strides=*/nullptr));

    // The full output width can be larger than this BRGEMM's N. For tiled
    // mm1, N is the kv-block width, but post-op descriptors cover the full
    // [M, seq_kv] score tile. At execution, data_C_ptr_ anchors that full tile
    // while ptr_D points at the current N-block; the binary injector uses
    // their difference to recover each element's row and global column.
    // This handles per-element and broadcast masks. The full output width
    // defaults to N when the full output fits one tile.
    const dim_t full_tile_width = full_output_width > 0 ? full_output_width : N;

    // Append the supported mm1 post-ops (scale / attention-mask) to the GEMM
    // post-op list. A scalar binary rhs is a [1 x 1] broadcast. For a tensor
    // rhs, axes with extent 1 stay 1 in the descriptor and broadcast; other
    // axes cover this tile and retain their user strides. The optional select
    // uses a scalar fill rhs and a dense [M x full_tile_width] condition.
    // Runtime pointers are supplied per execute call.
    const bool has_post_ops = post_ops && !post_ops->empty();
    if (has_post_ops || select_as_postop) {
        post_ops_t po;
        if (has_post_ops) {
            for (const auto &pop : *post_ops) {
                if (!pop.is_binary) {
                    CHECK(po.append_eltwise(
                            /*scale=*/1.0f, pop.alg, pop.alpha, pop.beta));
                    continue;
                }

                memory_desc_t rhs_md;
                if (pop.rhs_is_scalar) {
                    dims_t rhs_dims = {1, 1};
                    CHECK(memory_desc_init_by_tag(
                            rhs_md, 2, rhs_dims, pop.rhs_dt, format_tag::ab));
                } else {
                    const int row_axis = stack_heads ? sdpa_pd_t::mask_h_index
                                                     : sdpa_pd_t::mask_q_index;
                    const dim_t rows = pop.rhs_dims[row_axis] == 1 ? 1 : M;

                    constexpr int key_axis = sdpa_pd_t::mask_k_index;
                    const dim_t cols
                            = pop.rhs_dims[key_axis] == 1 ? 1 : full_tile_width;

                    dims_t rhs_dims = {rows, cols};
                    dims_t rhs_str = {pop.rhs_strides[row_axis],
                            pop.rhs_strides[key_axis]};

                    CHECK(memory_desc_init_by_strides(
                            rhs_md, 2, rhs_dims, pop.rhs_dt, rhs_str));
                }
                CHECK(po.append_binary(pop.alg, &rhs_md));
            }
        }
        if (select_as_postop) {
            memory_desc_t fill_md;
            dims_t fl_dims = {1, 1};
            CHECK(memory_desc_init_by_tag(
                    fill_md, 2, fl_dims, data_type::f32, format_tag::ab));

            memory_desc_t cond_md;
            dims_t cd_dims = {M, full_tile_width};
            CHECK(memory_desc_init_by_tag(
                    cond_md, 2, cd_dims, data_type::u8, format_tag::ab));

            CHECK(po.append_binary(
                    alg_kind::binary_select, &fill_md, &cond_md));
        }

        primitive_attr_t attr;
        CHECK(attr.set_post_ops(po));

        // LDC is the score tile's physical row stride. It equals full_tile_width
        // when an mm1 binary post-op has a tensor RHS (e.g., the attention
        // mask), or when select is folded into mm1 (see configure()).
        memory_desc_t dst_md;
        dims_t d_dims = {M, full_tile_width};
        dims_t d_strides = {ldc, 1};
        CHECK(memory_desc_init_by_strides(
                dst_md, 2, d_dims, data_type::f32, d_strides));

        CHECK(brgemm_desc_set_postops(&brg, &attr, &dst_md, /*LDD=*/ldc));
    }

    const bool amx_vnni_k_tail = (brg.is_bf16_tmm || brg.is_f16_tmm)
            && K % data_type_vnni_granularity(dt) != 0;
    // AMX follows the BRGEMM matmul kernel policy: SDPA always executes a
    // single batch element, which lets it use the AMX microkernel with
    // interleaved tile stores instead of the generic tile kernel.
    const bool use_amx_uker = brg.is_tmm;
    if (amx_vnni_k_tail || b_is_vnni || use_amx_uker) {
        brgemm_attr_t brgattr;
        brgattr.b_is_vnni = b_is_vnni;
        brgattr.extendable_k = amx_vnni_k_tail;
        brgattr.wary_A_k_tail_read = amx_vnni_k_tail;
        if (use_amx_uker) {
            brgattr.max_bs = 1;
            brgattr.use_uker = true;
            brgattr.use_interleave_stores = true;
        }
        CHECK(brgemm_desc_set_attr(&brg, brgattr));
    }

    CHECK(brgemm_desc_finalize(&brg));
    return status::success;
}

status_t is_vnni_b_required(
        data_type_t dt, bool attr_b_is_vnni, bool &needs_vnni_b) {
    brgemm_desc_t brg {};
    brg.dt_a = brg.dt_b = dt;
    CHECK(init_kernel_datatype(&brg, dt, dt));

    brgemm_utils::set_isa_impl(&brg);
    if (brg.isa_impl == isa_undef) return status::unimplemented;

    needs_vnni_b = brgemm_desc_t::is_b_data_layout_vnni(
            dt, dt, attr_b_is_vnni, brg.isa_impl);

    return status::success;
}

// Whether BRGEMM computes dt x dt products in AMX tiles on this CPU.
status_t uses_amx_tiles(data_type_t dt, bool &is_amx) {
    constexpr dim_t probe = 16;
    brgemm_desc_t brg;
    CHECK(brgemm_desc_init(&brg, isa_undef, brgemm_addr, dt, dt,
            /*transA=*/false, /*transB=*/false, brgemm_row_major,
            /*alpha=*/1.0f, /*beta=*/0.0f, probe, probe, probe, probe, probe,
            probe, /*strides=*/nullptr));
    is_amx = brg.is_tmm;
    return status::success;
}

} // namespace

sdpa_full_softmax_kernels_t::sdpa_full_softmax_kernels_t() = default;
sdpa_full_softmax_kernels_t::~sdpa_full_softmax_kernels_t() = default;

namespace sdpa_full_softmax {

status_t configure(const sdpa_full_softmax_params_t &params,
        sdpa_full_softmax_conf_t &conf) {
    conf.params = params;
    const sdpa_full_softmax_params_t &p_ = conf.params;

    if (!utils::one_of(
                p_.mm_dt, data_type::f32, data_type::f16, data_type::bf16))
        return status::unimplemented;

    if (!utils::one_of(
                p_.out_dt, data_type::f32, data_type::f16, data_type::bf16))
        return status::unimplemented;

    const dim_t mm_dt_sz = static_cast<dim_t>(types::data_type_size(p_.mm_dt));

    constexpr int ndims = sdpa_pd_t::ndims;
    constexpr int row_dim = ndims - 2;

    const dim_t k_seq_stride = p_.k_strides[ndims - 1];
    const dim_t k_hs_stride = p_.k_strides[row_dim];

    const bool mm1_transpose_k = k_seq_stride != 1;
    const bool transpose_v = p_.v_strides[ndims - 1] != 1;

    CHECK(is_vnni_b_required(p_.mm_dt, mm1_transpose_k, conf.mm1_needs_vnni_b));
    CHECK(is_vnni_b_required(p_.mm_dt, transpose_v, conf.mm2_needs_vnni_b));

    const bool need_copy_v = conf.mm2_needs_vnni_b || transpose_v;

    // Width of one packed K/V panel. A packed B operand is read by BRGEMM with
    // ldb = b_panel_n and an N block of at most b_panel_n (see b_packer_t).
    constexpr dim_t b_panel_n = sdpa_brgemm_copy::b_packer_t::panel_n;

    const dim_t l2_budget_bytes
            = static_cast<dim_t>(3 * platform::get_per_core_cache_size(2) / 4);

    bool is_amx = false;
    CHECK(uses_amx_tiles(p_.mm_dt, is_amx));
    conf.nthr = dnnl_get_max_threads();

    const dim_t num_head_kv = p_.num_head_q / p_.group_head;

    // Fold non-inverted select into mm1 when condition rows match dense score
    // rows. The pd guarantees a full, contiguous key axis. A single query row
    // does not use its row stride; otherwise require non-broadcast query rows
    // with stride seq_kv.
    conf.mm1_select_postop = false;
    if (p_.has_select) {
        const bool dense_query_axis = p_.seq_q == 1
                || (p_.cond_dims[sdpa_pd_t::mask_q_index] != 1
                        && p_.cond_strides[sdpa_pd_t::mask_q_index]
                                == p_.seq_kv);
        conf.mm1_select_postop = !p_.invert_select && dense_query_axis;
    }

    // The binary injector requires tensor rhs data to share the score tile's
    // layout. Keep score rows dense when a tensor rhs is used; scalar rhs and
    // eltwise post-ops are layout-independent.
    bool dense_scores_required = conf.mm1_select_postop;
    for (const auto &pop : p_.mm1_post_ops)
        if (pop.is_binary && !pop.rhs_is_scalar) dense_scores_required = true;
    conf.ld_scores
            = scratch_row_ld(p_.seq_kv, sizeof(float), !dense_scores_required);
    auto &thread_scratch = conf.thread_scratch;
    thread_scratch.set_compute_type(p_.mm_dt, mm_dt_sz);
    conf.ld_prob = thread_scratch.has_prob
            ? scratch_row_ld(p_.seq_kv, mm_dt_sz, /*allow_padding=*/true)
            : conf.ld_scores;

    // Choose the query block by trading K/V streaming against score-tile
    // traffic. Each work item streams all of K and V once, so their traffic per
    // score element falls as 1/q_block, while the live [q_block x seq_kv] score
    // tile (f32 scores plus the bf16/f16 P tile) grows with q_block and moves
    // from L2 to L3/memory. Larger heads amortize more K/V bytes per score, so
    // the score-tile target grows with the head size. A q_block x seq_kv x
    // head-size sweep on GNR (2 MiB L2) favored smaller score tiles for
    // bf16/AMX than for f32.
    const dim_t q_granule = is_amx ? 16 : 8; // AMX computes 16-row tiles
    const dim_t q_block_max = is_amx ? 96 : 192;
    const float score_tile_l2_ratio = is_amx ? 0.75f : 1.5f;
    const size_t score_row_bytes
            = thread_scratch.score_tile_row_bytes(conf.ld_scores, conf.ld_prob);
    // Empirical head-size scaling, normalized to 1 when both head sizes are 64.
    const float head_scale = std::sqrt((p_.hs_qk + p_.hs_v) / 128.f);
    const float score_tile_target_bytes = score_tile_l2_ratio * head_scale
            * platform::get_per_core_cache_size(2);
    const float q_rows
            = score_tile_target_bytes / static_cast<float>(score_row_bytes);
    // A long seq_kv shrinks the tile-based q_block until K/V streaming
    // dominates. Bound it so the K/V bytes per score element,
    // (hs_qk + hs_v) * mm_dt_sz / q_block, stay below the ~16..24 bytes of
    // score-tile traffic per element. Threads working on the same KV head at
    // the same time share its K/V through L3, which halves the bound once more
    // than two threads do.
    const dim_t kv_heads_total = p_.batch * num_head_kv;
    const bool kv_shared_in_l3 = 2 * kv_heads_total < conf.nthr;
    const float q_rows_min
            = (p_.hs_qk + p_.hs_v) / (kv_shared_in_l3 ? 16.f : 8.f);
    // Query rows of one work item: per-head q_block, or the rows of all heads
    // in a KV-head group.
    dim_t item_rows = q_granule
            * std::lround(nstl::max(q_rows, q_rows_min) / q_granule);
    item_rows = utils::saturate(q_granule, q_block_max, item_rows);

    // A KV-head group per work item reads each K/V tile once for all its query
    // heads instead of once per head. By default, use it when all query rows
    // of a group fit one work item and the groups alone fill the threads.
    using schedule_t = sdpa_full_softmax_schedule_t;
    conf.schedule = p_.schedule;
    if (conf.schedule == schedule_t::any) {
        const bool group_fits_item = p_.seq_q * p_.group_head <= item_rows;
        conf.schedule = p_.group_head > 1 && group_fits_item
                        && kv_heads_total >= conf.nthr
                ? schedule_t::kv_head_group
                : schedule_t::per_head;
    }
    conf.query_heads_per_item
            = conf.schedule == schedule_t::per_head ? 1 : p_.group_head;
    const bool split_kv = conf.schedule == schedule_t::split_kv;

    dim_t q_block = item_rows / conf.query_heads_per_item;
    if (q_block < p_.seq_q)
        q_block = nstl::max(q_granule, utils::rnd_dn(q_block, q_granule));
    q_block = nstl::min(q_block, p_.seq_q);
    // Shrink while that leaves fewer work items than threads.
    const dim_t head_items
            = p_.batch * p_.num_head_q / conf.query_heads_per_item;
    while (q_block > q_granule
            && head_items * utils::div_up(p_.seq_q, q_block) < conf.nthr)
        q_block = utils::rnd_dn(q_block - 1, q_granule);
    q_block = utils::saturate((dim_t)1, p_.seq_q, q_block);
    conf.q_block = q_block;
    const dim_t q_tail = p_.seq_q % q_block;
    conf.q_tail = q_tail;

    // Tile mm1's N axis (seq_kv) so each call's K panel [hs_qk x kv_block]
    // stays L2-resident; an untiled matmul re-streams its whole panel per
    // M-chunk and turns memory-bound once that panel exceeds ~L2. Size
    // kv_block so the panel fits its L2 budget, then round DOWN to a column
    // granule: kv_granule_large when the budget allows, else
    // kv_granule_medium, else kv_granule_min. A large head or a small L2 thus
    // shrinks the block rather than collapsing it to the whole (untiled) axis.
    // Clamp to seq_kv: short context runs as one untiled block. The granules
    // are a blocking heuristic, not a layout requirement: packed K accepts any
    // block (panel_offset() handles n % b_panel_n), and mm1 splits kv_block
    // into b_panel_n-wide calls on its own. A large granule of b_panel_n keeps
    // mm1's packed-K calls free of N tails. mm2's seq_kv blocking is chosen
    // separately below.
    constexpr dim_t kv_granule_large = b_panel_n;
    constexpr dim_t kv_granule_medium = 32;
    constexpr dim_t kv_granule_min = 16;
    const dim_t b_panel_budget_bytes = l2_budget_bytes / 8;
    dim_t kv_block = b_panel_budget_bytes / (p_.hs_qk * mm_dt_sz);
    const dim_t granule = kv_block >= kv_granule_large
            ? kv_granule_large
            : (kv_block >= kv_granule_medium ? kv_granule_medium
                                             : kv_granule_min);
    kv_block = nstl::max<dim_t>(
            utils::rnd_dn(kv_block, granule), kv_granule_min);
    kv_block = nstl::min<dim_t>(kv_block, p_.seq_kv);

    // An f32 K already laid out as [hs_qk, seq_kv] is read in place by mm1 as
    // [hs_qk x kv_block] B panels. If the K row stride maps many panel rows to
    // the same L2 sets, the panel is evicted between re-reads, so it is packed
    // into dense b_panel_n-wide panels instead. The panel shares L2 with the
    // score tile, Q, the V panel and prefetched lines, so it starts thrashing well
    // before it alone fills a set. Packing costs one extra pass over K per KV
    // head and only pays off when the panel is re-read by enough query rows
    // (seq_q * group_head): moderate aliasing needs more reuse to amortize it
    // than a panel that alone overflows a set.
    const dim_t k_reuse_rows = p_.seq_q * p_.group_head;
    const float k_l2_set_load
            = f32_tile_l2_set_load(p_.hs_qk, kv_block, k_hs_stride);
    const bool pack_k_for_aliasing = p_.mm_dt == data_type::f32
            && !mm1_transpose_k
            && ((k_reuse_rows >= 256 && k_l2_set_load > 0.5f)
                    || (k_reuse_rows >= 128 && k_l2_set_load >= 1.0f));
    const bool need_copy_k
            = mm1_transpose_k || conf.mm1_needs_vnni_b || pack_k_for_aliasing;
    conf.pack_k_for_aliasing = pack_k_for_aliasing;
    conf.pack_k = need_copy_k;
    conf.pack_v = need_copy_v;

    // Packing per tile removes the separate packing pass (a round trip of the
    // packed K/V through memory, parallel over batch x KV heads only), but
    // repeats the packing in every work item that reads the tile. By default,
    // pack per tile when a work item covers all query rows of its heads: then
    // only the heads of one KV-head group re-pack a tile, and they run at the
    // same time, so the re-reads hit the shared cache. With split_kv, every
    // K/V tile is read by exactly one task, so packing per tile never repeats.
    using kv_packing_t = sdpa_full_softmax_kv_packing_t;
    conf.kv_packing = p_.kv_packing;
    if (conf.kv_packing == kv_packing_t::any)
        conf.kv_packing = split_kv || conf.q_block == p_.seq_q
                ? kv_packing_t::per_tile
                : kv_packing_t::ahead;
    const bool pack_per_tile = conf.kv_packing == kv_packing_t::per_tile;

    // Packed K is laid out in independent b_panel_n-column panels, which limits
    // mm1's N block to one panel. mm2 only reduces over seq_kv after the
    // softmax is complete, so its K block is chosen independently and kept
    // deep: a reduction as shallow as mm1's N block would split it into many
    // short calls, each re-reading and re-writing the f32 [m x hs_v] partial
    // output. Packed V limits each call's B panel to [mm2_k_block x v_block]
    // with v_block <= b_panel_n, so hs_v does not bound the block depth. On
    // GNR, a 256-deep AMX block was best or close for head sizes 64..256, and
    // on GNR, SPR and DMR it matched or beat the old head-size-bounded block
    // (64..192 deep) for head sizes up to 1024, including asymmetric ones;
    // for f32, 512..1024-deep blocks performed best for head sizes 64..256
    // (GNR).
    constexpr dim_t amx_mm2_k_block = 256;
    constexpr dim_t mm2_k_block = 512;
    conf.mm1_n_block
            = need_copy_k ? nstl::min<dim_t>(kv_block, b_panel_n) : kv_block;
    conf.mm1_n_tail = p_.seq_kv % conf.mm1_n_block;
    conf.mm2_k_block
            = nstl::min(p_.seq_kv, is_amx ? amx_mm2_k_block : mm2_k_block);
    conf.mm2_k_tail = p_.seq_kv % conf.mm2_k_block;
    // >1 block -> mm2 accumulates across blocks: the first block uses a beta = 0
    // kernel (fresh C) and the rest beta = 1, so no destination pre-zeroing is
    // needed. A single block keeps the original fresh-C (beta = 0) path.
    const float mm2_beta = conf.mm2_k_block < p_.seq_kv ? 1.0f : 0.0f;
    conf.mm2_beta = mm2_beta;

    // split_kv splits each work item's seq_kv range into about as many chunks
    // as needed to give every thread a task, but not shorter than
    // min_chunk_keys, so that each task amortizes its setup. A chunk is a whole
    // number of blocks and the last chunk also takes the remainder, so every
    // chunk starts with a full block: the block/tail descriptors stay valid,
    // and mm2 starts every chunk with its beta = 0 kernel.
    if (split_kv) {
        constexpr dim_t min_chunk_keys = 256;
        const dim_t items = head_items * utils::div_up(p_.seq_q, q_block);
        const dim_t chunks_per_item = utils::div_up(conf.nthr, items);
        auto split = [&](dim_t block, dim_t &chunk, dim_t &chunks) {
            chunk = utils::rnd_up(
                    nstl::max(min_chunk_keys,
                            utils::div_up(p_.seq_kv, chunks_per_item)),
                    block);
            chunks = nstl::max<dim_t>(1, p_.seq_kv / chunk);
        };
        auto &sk = conf.split_kv;
        split(conf.mm1_n_block, sk.mm1_chunk, sk.mm1_chunks);
        split(conf.mm2_k_block, sk.mm2_chunk, sk.mm2_chunks);
    }

    // mm2 can write its f32 result straight into the user output tensor (C =
    // output, LDC = output row stride), skipping the pv scratch tile and the
    // down-convert/scatter, when the output is f32 with unit-stride rows. All
    // other cases (bf16/f16 output or a strided output column) keep pv+scatter.
    // split_kv sums the results of the mm2 chunks before writing the output.
    const dim_t o_col_stride = p_.o_strides[ndims - 1];
    const bool mm2_direct
            = !split_kv && p_.out_dt == data_type::f32 && o_col_stride == 1;
    conf.mm2_direct = mm2_direct;
    conf.ld_pv = scratch_row_ld(p_.hs_v, sizeof(float), /*allow_padding=*/true);

    // With seq_q == 1, the queries in a kv_head_group work item come from
    // different heads and can be processed together as BRGEMM rows.
    // The per-head score/P/pv tiles of a work item are already consecutive
    // [1 x ld] rows. Every row-addressed operand must then use its head stride
    // as the row stride:
    //  - Q and a direct output: head rows must not overlap;
    //  - a tensor mm1 RHS (mask) and a folded select condition: the binary
    //    injector addresses them with score-tile offsets, so they must either
    //    broadcast over heads or have a head stride equal to ld_scores.
    constexpr int head_dim = 1;
    auto can_stack_heads = [&]() {
        if (!p_.allow_stack_heads || p_.seq_q != 1
                || conf.query_heads_per_item == 1)
            return false;
        if (p_.q_strides[head_dim] < p_.hs_qk) return false;
        if (mm2_direct && p_.o_strides[head_dim] < p_.hs_v) return false;
        for (const auto &pop : p_.mm1_post_ops) {
            if (!pop.is_binary || pop.rhs_is_scalar) continue;
            if (pop.rhs_dims[head_dim] != 1
                    && pop.rhs_strides[head_dim] != conf.ld_scores)
                return false;
        }
        if (conf.mm1_select_postop
                && (p_.cond_dims[head_dim] == 1
                        || p_.cond_strides[head_dim] != conf.ld_scores))
            return false;
        return true;
    };

    // Resolve the BRGEMM row layout once; descriptors and post-ops below
    // are built from it.
    conf.stack_heads = can_stack_heads();
    const int m_axis = conf.stack_heads ? head_dim : row_dim;
    conf.m_block = conf.stack_heads ? conf.query_heads_per_item : q_block;
    conf.mm1_lda = p_.q_strides[m_axis];
    conf.out_ld = p_.o_strides[m_axis];
    conf.mm2_ldc = mm2_direct ? conf.out_ld : conf.ld_pv;

    // A/B leading dims come from the user strides; scores/pv are dense.
    //   mm1: scores[m, seq_kv] = Q[m, hs_qk] * K[hs_qk, seq_kv]
    //   mm2: pv[m, hs_v]       = P[m, seq_kv] * V[seq_kv, hs_v]
    // The brgemm ukernel does not support a transposed B operand and needs mm1
    // B row-major as [hs_qk, seq_kv] (seq_kv unit-stride). Decide the transpose
    // from the K tensor's PHYSICAL strides, not the graph transpose_b attr:
    //   * transpose_b == true  : K is [.., seq_kv, hs_qk] -> seq_kv axis is
    //                            row_dim, hs_qk axis is the last dim.
    //   * transpose_b == false : K is [.., hs_qk, seq_kv] -> hs_qk axis is
    //                            row_dim, seq_kv axis is the last dim.
    // In either orientation, if the seq_kv axis is not unit-stride a dense
    // [hs_qk, seq_kv] transpose is materialised once (see execute); a K that is
    // logically pre-transposed but physically stored with hs_qk contiguous
    // therefore still gets transposed here.
    // mm1 materialises a dense (and, for bf16/f16, VNNI-packed) [hs_qk, seq_kv]
    // B operand whenever it must transpose K or pack it, laid out in
    // b_panel_n-wide panels: ldb is then b_panel_n and one call covers at most
    // one panel. Only the plain-f32, already-[hs_qk, seq_kv] case reads K in
    // place. V is handled the same way, so v_block is one panel when packed.
    conf.ldk = need_copy_k ? b_panel_n : p_.k_strides[row_dim];
    conf.ldv = need_copy_v ? b_panel_n : p_.v_strides[row_dim];
    conf.v_block = need_copy_v ? nstl::min<dim_t>(p_.hs_v, b_panel_n) : p_.hs_v;
    conf.v_tail = p_.hs_v % conf.v_block;

    // Build mm1/mm2 descriptors and size tile-store scratch for the largest
    // workspace. The descriptor container is shared across pd clones.
    auto &brg_conf = conf.brgemm;
    brg_conf = {};
    brg_conf.descs
            = std::make_shared<brgemm_containers::brgemm_desc_container_t>();
    size_t max_wsp = 0;
    auto add_brg_desc
            = [&](int &idx, brgemm_desc_t &brg, bool needs_vnni) -> status_t {
        // AMX ukernels require VNNI-packed B.
        if (brg.is_tmm && !needs_vnni) return status::unimplemented;
        if (brg.is_tmm) {
            // The palette container's insert() cannot distinguish a duplicate
            // palette from an initialization failure, so validate it here.
            char palette[AMX_PALETTE_SIZE];
            CHECK(brgemm_init_tiles(brg, palette));
        }
        max_wsp = nstl::max(
                max_wsp, static_cast<size_t>(brg.get_wsp_buffer_size()));
        idx = brg_conf.descs->insert(brg);
        return status::success;
    };

    // One descriptor per block/tail combination: rows split into m_block
    // (+ q_tail); seq_kv splits into mm1_n_block (+ mm1_n_tail) for mm1 and
    // independently into mm2_k_block (+ mm2_k_tail) for mm2.
    const dim_t ms[2] = {conf.m_block, conf.q_tail};

    // mm1 N is one seq_kv column block; its post-op descriptors keep the full
    // seq_kv width (full_output_width) so the folded mask/select is addressed
    // by global column.
    auto build_mm1_descs = [&]() -> status_t {
        const dim_t ns[2] = {conf.mm1_n_block, conf.mm1_n_tail};
        for (int qi = 0; qi < 2; ++qi) {
            if (!ms[qi]) continue;
            for (int ni = 0; ni < 2; ++ni) {
                if (!ns[ni]) continue;
                brgemm_desc_t brg;
                CHECK(build_brgemm_desc(brg, p_.mm_dt,
                        /*beta=*/0.0f, ms[qi], ns[ni], p_.hs_qk,
                        /*lda=*/conf.mm1_lda,
                        /*ldb=*/conf.ldk,
                        /*ldc=*/conf.ld_scores, &p_.mm1_post_ops,
                        conf.mm1_select_postop,
                        /*full_output_width=*/p_.seq_kv,
                        /*b_is_vnni=*/conf.mm1_needs_vnni_b, conf.stack_heads));
                CHECK(add_brg_desc(
                        brg_conf.mm1[qi][ni], brg, conf.mm1_needs_vnni_b));
            }
        }
        return status::success;
    };

    // mm2 K is one seq_kv reduction block; beta accumulates across them. B is
    // V in place when its columns are contiguous and no VNNI packing is needed;
    // otherwise it uses the materialized V buffer. The beta = 0 full-K-block
    // mm2 for the first block (when the reduction is tiled) is built per
    // q-tile.
    auto build_mm2_descs = [&]() -> status_t {
        const dim_t ks[2] = {conf.mm2_k_block, conf.mm2_k_tail};
        for (int qi = 0; qi < 2; ++qi) {
            if (!ms[qi]) continue;
            for (int ki = 0; ki < 2; ++ki) {
                if (!ks[ki]) continue;
                for (int vi = 0; vi < 2; ++vi) {
                    const dim_t width = vi ? conf.v_tail : conf.v_block;
                    if (!width) continue;
                    brgemm_desc_t brg;
                    CHECK(build_brgemm_desc(brg, p_.mm_dt, mm2_beta, ms[qi],
                            width, ks[ki],
                            /*lda=*/conf.ld_prob, /*ldb=*/conf.ldv,
                            /*ldc=*/conf.mm2_ldc, nullptr, false, 0,
                            conf.mm2_needs_vnni_b));
                    CHECK(add_brg_desc(brg_conf.mm2[qi][ki][vi], brg,
                            conf.mm2_needs_vnni_b));
                }
            }
            if (mm2_beta != 0.0f) {
                for (int vi = 0; vi < 2; ++vi) {
                    const dim_t width = vi ? conf.v_tail : conf.v_block;
                    if (!width) continue;
                    brgemm_desc_t brg;
                    CHECK(build_brgemm_desc(brg, p_.mm_dt, 0.0f, ms[qi], width,
                            conf.mm2_k_block, conf.ld_prob, conf.ldv,
                            conf.mm2_ldc, nullptr, false, 0,
                            conf.mm2_needs_vnni_b));
                    CHECK(add_brg_desc(brg_conf.mm2_beta0[qi][vi], brg,
                            conf.mm2_needs_vnni_b));
                }
            }
        }
        return status::success;
    };

    CHECK(build_mm1_descs());
    CHECK(build_mm2_descs());

    // Per-thread scratch, per query head of the work item: one score tile
    // [q_block x seq_kv] and one pv tile [q_block x hs_v], both f32 (BRGEMM
    // accumulates in f32). For a non-f32 compute type, a third
    // [q_block x seq_kv] tile holds P (mm2's A operand) down-converted from the
    // f32 softmax output; f32 reuses the scores tile directly. The pv tile lets
    // mm2 write a dense output that is then scattered (and down-converted) to
    // the (possibly strided) user output. The tiles of the work item's heads
    // are consecutive. split_kv keeps these tiles in shared buffers instead.
    const size_t item_q_rows = split_kv
            ? 0
            : static_cast<size_t>(conf.query_heads_per_item) * q_block;
    conf.amx_wsp_bytes = align64(max_wsp);
    thread_scratch.set_tile_buffers(item_q_rows, conf.ld_scores, conf.ld_prob,
            conf.ld_pv, mm2_direct, conf.amx_wsp_bytes);
    auto &ts = thread_scratch;

    // Packed K and V buffers: per thread for one tile each when packing per
    // tile, else whole tensors placed consecutively in the shared area.
    using b_packer_t = sdpa_brgemm_copy::b_packer_t;
    auto make_b_copy_desc = [&](size_t scratch_offset, dim_t rows, dim_t cols,
                                    bool needs_vnni) {
        tensor_copy_desc_t copy_desc;
        copy_desc.scratch_offset = scratch_offset;
        copy_desc.padded_k = b_packer_t::packed_k(rows, p_.mm_dt);
        copy_desc.vnni_factor
                = needs_vnni ? data_type_vnni_granularity(p_.mm_dt) : 1;
        copy_desc.head_stride = static_cast<size_t>(copy_desc.padded_k)
                * b_packer_t::packed_ld(cols) * mm_dt_sz;
        copy_desc.batch_stride
                = static_cast<size_t>(num_head_kv) * copy_desc.head_stride;
        return copy_desc;
    };
    size_t shared_offset = 0;
    if (pack_per_tile) {
        // A K tile is one mm1 N block (at most one panel); a V tile is one mm2
        // K block across all of hs_v.
        if (need_copy_k) {
            conf.k_tile_desc = make_b_copy_desc(
                    ts.size, p_.hs_qk, conf.mm1_n_block, conf.mm1_needs_vnni_b);
            ts.size += align64(conf.k_tile_desc->head_stride);
        }
        if (need_copy_v) {
            conf.v_tile_desc = make_b_copy_desc(
                    ts.size, conf.mm2_k_block, p_.hs_v, conf.mm2_needs_vnni_b);
            ts.size += align64(conf.v_tile_desc->head_stride);
        }
    } else {
        if (need_copy_k) {
            conf.k_copy_desc = make_b_copy_desc(
                    shared_offset, p_.hs_qk, p_.seq_kv, conf.mm1_needs_vnni_b);
            shared_offset += static_cast<size_t>(p_.batch)
                    * conf.k_copy_desc->batch_stride;
        }
        if (need_copy_v) {
            conf.v_copy_desc = make_b_copy_desc(
                    shared_offset, p_.seq_kv, p_.hs_v, conf.mm2_needs_vnni_b);
            shared_offset += static_cast<size_t>(p_.batch)
                    * conf.v_copy_desc->batch_stride;
        }
    }
    if (split_kv) {
        auto &sk = conf.split_kv;
        const size_t rows
                = static_cast<size_t>(p_.batch) * p_.num_head_q * p_.seq_q;
        auto add_shared_buffer = [&](size_t bytes) {
            const size_t offset = align64(shared_offset);
            shared_offset = offset + bytes;
            return offset;
        };
        sk.scores = add_shared_buffer(rows * conf.ld_scores * sizeof(float));
        sk.prob = ts.has_prob
                ? add_shared_buffer(rows * conf.ld_prob * mm_dt_sz)
                : sk.scores;
        sk.partials = add_shared_buffer(
                sk.mm2_chunks * rows * conf.ld_pv * sizeof(float));
    }
    conf.scratch_per_thread = ts.size;
    conf.shared_scratch_bytes = shared_offset;

    return status::success;
}

status_t create_kernels(cpu_isa_t isa_for_ir,
        const sdpa_full_softmax_conf_t &conf, engine_t *engine,
        sdpa_full_softmax_kernels_t &kernels) {
    const sdpa_full_softmax_params_t &p_ = conf.params;
    const dim_t q_block_ = conf.q_block;
    auto &softmax_pd_ = kernels.softmax_pd;
    auto &softmax_kernel_ = kernels.softmax_kernel;
    auto &select_kernel_ = kernels.select_kernel;

    const dim_t seq_kv = p_.seq_kv;
    constexpr int ndims = sdpa_pd_t::ndims;
    constexpr int row_dim = ndims - 2;

    // Packers for B = [rows x cols] with the given source strides. Packing ahead
    // packs the whole tensor; packing per tile packs one full block of the
    // tiled axis ([0]) or its tail ([1]).
    auto init_packers =
            [&](std::unique_ptr<sdpa_brgemm_copy::b_packer_t>(&packers)[2],
                    const dim_t(&rows)[2], const dim_t(&cols)[2],
                    dim_t row_stride, dim_t col_stride, dim_t ld) -> status_t {
        for (int i = 0; i < 2; ++i) {
            if (!rows[i] || !cols[i]) continue;
            packers[i].reset(new sdpa_brgemm_copy::b_packer_t());
            CHECK(packers[i]->init(rows[i], cols[i], row_stride, col_stride, ld,
                    p_.mm_dt, /*separate_panels=*/true));
        }
        return status::success;
    };
    const bool per_tile
            = conf.kv_packing == sdpa_full_softmax_kv_packing_t::per_tile;
    if (conf.pack_k) {
        const dim_t rows[2] = {p_.hs_qk, p_.hs_qk};
        const dim_t cols[2] = {per_tile ? conf.mm1_n_block : seq_kv,
                per_tile ? conf.mm1_n_tail : 0};
        CHECK(init_packers(kernels.k_packers, rows, cols, p_.k_strides[row_dim],
                p_.k_strides[ndims - 1], conf.ldk));
    }
    if (conf.pack_v) {
        const dim_t rows[2] = {per_tile ? conf.mm2_k_block : seq_kv,
                per_tile ? conf.mm2_k_tail : 0};
        const dim_t cols[2] = {p_.hs_v, p_.hs_v};
        CHECK(init_packers(kernels.v_packers, rows, cols, p_.v_strides[row_dim],
                p_.v_strides[ndims - 1], conf.ldv));
    }

    const auto &descs = conf.brgemm.descs;
    const int num_descs = static_cast<int>(descs->refs_size());
    kernels.brgemm_kernels.resize(num_descs);
    kernels.brgemm_palettes.resize(num_descs);
    for (int idx = 0; idx < num_descs; ++idx) {
        // The descriptor container may have unused entries after growing.
        const auto *brg = (*descs)[idx];
        if (!brg) continue;
        CHECK(kernels.brgemm_kernels.insert(idx, brg));
        if (brg->is_tmm) kernels.brgemm_palettes.insert(idx, brg);
    }

    // Reuse the vectorized jit softmax kernel for the max/exp/normalize over
    // the seq_kv axis. For xf16 mm2, keep the src f32 and write the normalized
    // result directly in mm_dt, avoiding a separate conversion pass. The
    // kernel is per-row, so the same instance serves the query-tail block too.
    {
        memory_desc_t sm_src_md, sm_dst_md;
        dims_t sm_dims = {q_block_, seq_kv};
        CHECK(memory_desc_init_by_tag(
                sm_src_md, 2, sm_dims, data_type::f32, format_tag::ab));
        CHECK(memory_desc_init_by_tag(
                sm_dst_md, 2, sm_dims, p_.mm_dt, format_tag::ab));
        softmax_desc_t sd {};
        sd.primitive_kind = primitive_kind::softmax;
        sd.prop_kind = prop_kind::forward_inference;
        // inf_as_zero: a fully-masked row (all -inf) must produce an
        // all-zero row instead of NaN. The jit softmax kernel implements
        // this natively for the softmax_accurate_inf_as_zero alg.
        sd.alg_kind = p_.softmax_inf_as_zero
                ? alg_kind::softmax_accurate_inf_as_zero
                : alg_kind::softmax_accurate;
        sd.src_desc = sm_src_md;
        sd.dst_desc = sm_dst_md;
        sd.softmax_axis = 1;

        primitive_attr_t attr;
        primitive_desc_iterator_t it(engine,
                reinterpret_cast<const op_desc_t *>(&sd), &attr, nullptr);
        VCONDCHECK(primitive, create, dispatch, sdpa, it.is_initialized(),
                status::unimplemented, VERBOSE_PRIMITIVE_CREATION_FAIL,
                "softmax");
        // Walk the dispatched impls (order-independent) for the jit softmax pd;
        // its kernel is extracted below and called directly per row.
        jit_uni_softmax_fwd_t::pd_t *jit_pd = nullptr;
        while (++it != it.end()) {
            if (auto *p = dynamic_cast<jit_uni_softmax_fwd_t::pd_t *>(
                        (*it).get())) {
                softmax_pd_ = *it;
                jit_pd = p;
                break;
            }
        }
        VCONDCHECK(primitive, create, dispatch, sdpa, jit_pd != nullptr,
                status::unimplemented, VERBOSE_PRIMITIVE_CREATION_FAIL,
                "softmax");
        softmax_impl::jit_softmax_kernel_base_t *k
                = softmax_impl::jit_softmax_kernel_base_t::create(jit_pd,
                        jit_pd->isa_, jit_pd->axis_is_plain_and_strided_);
        if (!k) return status::unimplemented;
        status_t st = k->create_kernel();
        if (st != status::success) {
            delete k;
            return st;
        }
        softmax_kernel_.reset(k);
    }

    // Build the standalone select-mask pre-pass kernel when a select mask is
    // present and is not already folded into mm1. Polarity, the seq_kv width,
    // and the score/condition row strides are baked in; the row count is a
    // runtime argument, so one instance serves both the full and query-tail
    // tiles. The condition is dense along seq_kv (cond_col == 1) because the
    // pd requires the key axis to be full and plain.
    if (p_.has_select && !conf.mm1_select_postop) {
        const dim_t cond_row = p_.cond_dims[sdpa_pd_t::mask_q_index] == 1
                ? 0
                : p_.cond_strides[sdpa_pd_t::mask_q_index];
        std::shared_ptr<jit_generator_t> k(
                sdpa_full_softmax_select_ir::create_select_ir_kernel(isa_for_ir,
                        /*wseq_kv=*/seq_kv,
                        /*invert_select=*/p_.invert_select,
                        /*scores_row_stride=*/conf.ld_scores,
                        /*cond_row_stride=*/cond_row));
        CHECK(k->create_kernel());
        select_kernel_ = std::move(k);
    }

    return status::success;
}

namespace {

// Inputs of one execute() call, shared by all of its tasks.
struct exec_state_t {
    exec_state_t(const sdpa_full_softmax_conf_t &conf,
            const sdpa_full_softmax_kernels_t &kernels,
            const sdpa_full_softmax_run_args_t &args, int nthr)
        : conf(conf)
        , p(conf.params)
        , kernels(kernels)
        , args(args)
        , mm_dt_sz(types::data_type_size(conf.params.mm_dt))
        , scratch(static_cast<char *>(args.scratch))
        , shared_area_offset(conf.scratch_per_thread * nthr)
        , q_td(get_tensor_desc(
                  args.q, conf.q_copy_desc, scratch, shared_area_offset))
        , k_td(get_tensor_desc(
                  args.k, conf.k_copy_desc, scratch, shared_area_offset))
        , v_td(get_tensor_desc(
                  args.v, conf.v_copy_desc, scratch, shared_area_offset))
        , cond_td(get_tensor_desc(
                  args.cond, conf.cond_copy_desc, scratch, shared_area_offset))
        , o_td(get_tensor_desc(
                  args.out, conf.o_copy_desc, scratch, shared_area_offset)) {}

    char *thread_scratch(int tid) const {
        return scratch + tid * conf.scratch_per_thread;
    }
    char *shared_area() const { return scratch + shared_area_offset; }

    // Select condition rows of a query head from query q0, or null without a
    // select.
    const uint8_t *cond(dim_t batch_idx, dim_t head, dim_t q0) const {
        if (!p.has_select) return nullptr;
        return reinterpret_cast<const uint8_t *>(
                get_arg_ptr(cond_td, batch_idx, head, q0));
    }

    const sdpa_full_softmax_conf_t &conf;
    const sdpa_full_softmax_params_t &p;
    const sdpa_full_softmax_kernels_t &kernels;
    const sdpa_full_softmax_run_args_t &args;
    const size_t mm_dt_sz;
    char *const scratch;
    const size_t shared_area_offset;
    const tensor_desc_t<const char> q_td, k_td, v_td, cond_td;
    const tensor_desc_t<char> o_td;
};

// When packing ahead, materialise mm1's B (K as a dense [hs_qk, seq_kv]
// matrix; the brgemm ukernel has no transposed-B mode) and mm2's B (V) once
// per (batch, kv_head) into the shared area. bf16/f16 are VNNI-packed over the
// reduction axis.
void pack_ahead(const exec_state_t &st) {
    const dim_t num_head_kv = st.p.num_head_q / st.p.group_head;
    auto pack = [&](const tensor_desc_t<const char> &td,
                        const sdpa_brgemm_copy::b_packer_t &packer) {
        parallel_nd(st.p.batch, num_head_kv,
                [&](dim_t batch_idx, dim_t kv_head_idx) {
            packer.execute(get_arg_ptr(td, batch_idx, kv_head_idx),
                    get_copy_ptr(td, batch_idx, kv_head_idx));
        });
    };
    if (st.k_td.copy_desc) pack(st.k_td, *st.kernels.k_packers[0]);
    if (st.v_td.copy_desc) pack(st.v_td, *st.kernels.v_packers[0]);
}

// Score, P or f32 mm2-result tiles of the query heads of a work item: query
// head h of the item uses the tile at base + h * head_stride bytes.
struct item_tiles_t {
    char *base;
    size_t head_stride;

    char *of(dim_t h) const { return base + h * head_stride; }
    float *f32(dim_t h) const { return reinterpret_cast<float *>(of(h)); }
};

// Full-axis softmax: scores[n_rows, seq_kv] -> P[n_rows, seq_kv], normalized
// over all keys independently for each query row. The jit softmax kernel does
// the max/exp/normalize; for xf16 it writes the normalized result directly to
// P while using scores for its f32 interim. A select not folded into mm1 is
// applied in an IR pre-pass first; its condition is dense along seq_kv, and
// both polarities and the broadcast-over-rows case are baked in at build time.
void softmax_rows(const exec_state_t &st, float *scores, char *prob,
        const uint8_t *cond, dim_t n_rows) {
    const auto &conf = st.conf;
    if (cond && !conf.mm1_select_postop) {
        sdpa_full_softmax_select_ir::select_row_args_t sa;
        sa.scores = scores;
        sa.cond = cond;
        sa.fill = &st.args.fill;
        sa.n_rows = n_rows;
        (*st.kernels.select_kernel)(&sa);
    }
    const bool is_f32 = st.p.mm_dt == data_type::f32;
    for (dim_t i = 0; i < n_rows; ++i) {
        float *srow = scores + i * conf.ld_scores;
        void *prob_row = prob + i * conf.ld_prob * st.mm_dt_sz;
        softmax_impl::jit_softmax_kernel_base_t::call_params_t sp;
        sp.src = srow;
        sp.dst = prob_row;
        sp.diff_dst = nullptr;
        sp.interim = is_f32 ? nullptr : srow;
        sp.src_scales = nullptr;
        sp.dst_scales = nullptr;
        sp.process_n_elems = static_cast<size_t>(st.p.seq_kv);
        sp.dst_orig = prob_row;
        sp.post_ops_binary_rhs_arg_vec = nullptr;
        (*st.kernels.softmax_kernel)(&sp);
    }
}

// One work item: query block q_block_idx of the query_heads_per_item
// consecutive query heads starting at item_head_idx * query_heads_per_item,
// which share one KV head. Its operations take a key range and the tiles to
// use, so a work item serves both a whole fused computation in per-thread
// tiles and one split_kv chunk in shared tiles. AMX tiles configured by its
// BRGEMM calls are released on destruction.
class work_item_t {
public:
    work_item_t(const exec_state_t &st, int tid, dim_t batch_idx,
            dim_t item_head_idx, dim_t q_block_idx)
        : st_(st)
        , conf_(st.conf)
        , batch_idx_(batch_idx)
        , kv_head_idx_(
                  item_head_idx * conf_.query_heads_per_item / st.p.group_head)
        , m_(nstl::min(conf_.q_block, st.p.seq_q - q_block_idx * conf_.q_block))
        , qi_(m_ != conf_.q_block)
        , call_m_(qi_ ? conf_.q_tail : conf_.m_block)
        // Stacked heads: one BRGEMM call covers all heads of the work item,
        // with head 0's pointers and the head stride as the row stride.
        , brgemm_calls_(conf_.stack_heads ? 1 : conf_.query_heads_per_item)
        , has_mm1_postops_(
                  !st.p.mm1_post_ops.empty() || conf_.mm1_select_postop)
        , thread_scratch_(st.thread_scratch(tid))
        , amx_wsp_(conf_.amx_wsp_bytes > 0
                          ? thread_scratch_ + conf_.thread_scratch.amx_wsp
                          : nullptr)
        , heads_(conf_.query_heads_per_item) {
        const dim_t head0 = item_head_idx * conf_.query_heads_per_item;
        const dim_t q0 = q_block_idx * conf_.q_block;
        for (dim_t h = 0; h < conf_.query_heads_per_item; ++h) {
            const dim_t head = head0 + h;
            auto &hp = heads_[h];
            hp.q = get_arg_ptr(st.q_td, batch_idx, head, q0);
            hp.out = get_arg_ptr(st.o_td, batch_idx, head, q0);
            hp.cond = st.cond(batch_idx, head, q0);
            if (has_mm1_postops_)
                append_mm1_rhs(rhs_, st.p, st.args, batch_idx, head, q0,
                        conf_.mm1_select_postop, hp.cond);
        }
        rhs_per_head_ = rhs_.size() / conf_.query_heads_per_item;
    }

    ~work_item_t() {
        // Do not leak the configured AMX state past this work item.
        if (used_amx_) amx_tile_release();
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(work_item_t);

    // The user output of the item's query heads, for mm2 writing to it
    // directly.
    item_tiles_t output_tiles() const {
        const size_t head_stride
                = st_.p.o_strides[1] * types::data_type_size(st_.p.out_dt);
        return {heads_[0].out, head_stride};
    }

    // mm1: scores[m, kv_begin:kv_end) = Q[m, hs_qk] * K[hs_qk, kv_begin:kv_end)
    // in mm1_n_block-wide column blocks (+ tail). Each call writes its columns
    // of the full score tile (LDC = ld_scores). The post-op injector addresses
    // the mask or select condition at the global column: data_C_ptr_ is the
    // full-tile base, while c_blk is offset by the column.
    void mm1(dim_t kv_begin, dim_t kv_end, const item_tiles_t &scores) {
        const auto &brg = conf_.brgemm;
        b_operand_t k_op(st_.k_td, batch_idx_, kv_head_idx_,
                st_.kernels.k_packers, conf_.k_tile_desc, thread_scratch_,
                st_.mm_dt_sz);
        for_each_block(kv_begin, kv_end, conf_.mm1_n_block,
                [&](dim_t kv0, bool n_tail) {
            k_op.load_tile(0, kv0, n_tail);
            const brgemm_kernel_t *kernel = this->kernel(brg.mm1[qi_][n_tail]);
            for (dim_t h = 0; h < brgemm_calls_; ++h) {
                brgemm_batch_element_t be;
                be.ptr.A = heads_[h].q;
                be.ptr.B = k_op.at(0, kv0);
                float *c_blk = scores.f32(h) + kv0;
                if (has_mm1_postops_) {
                    brgemm_post_ops_data_t pod(/*bias=*/nullptr,
                            /*binary_post_ops_rhs=*/rhs_.data()
                                    + h * rhs_per_head_,
                            /*oc_logical_off=*/0,
                            /*dst_row_logical_off=*/0,
                            /*data_C_ptr_=*/scores.of(h),
                            /*first_mb_matrix_addr_off=*/0);
                    brgemm_kernel_execute_postops(
                            kernel, 1, &be, c_blk, c_blk, pod, amx_wsp_);
                } else {
                    brgemm_kernel_execute(kernel, 1, &be, c_blk, amx_wsp_);
                }
            }
        });
    }

    // Softmax of the item's score rows; see softmax_rows().
    void softmax(const item_tiles_t &scores, const item_tiles_t &prob) const {
        for (dim_t h = 0; h < conf_.query_heads_per_item; ++h)
            softmax_rows(st_, scores.f32(h), prob.of(h), heads_[h].cond, m_);
    }

    // mm2: c[m, hs_v] = P[m, kv_begin:kv_end) * V[kv_begin:kv_end, hs_v], tiled
    // over the reduction so each call's B panel [mm2_k_block, v_block] is
    // L2-resident. kv_begin must start a full block. With more than one
    // reduction block in seq_kv, the first block of the range uses a beta = 0
    // kernel (fresh C) and the rest beta = 1, so no destination pre-zeroing is
    // needed.
    void mm2(dim_t kv_begin, dim_t kv_end, const item_tiles_t &prob,
            const item_tiles_t &c) {
        const auto &brg = conf_.brgemm;
        b_operand_t v_op(st_.v_td, batch_idx_, kv_head_idx_,
                st_.kernels.v_packers, conf_.v_tile_desc, thread_scratch_,
                st_.mm_dt_sz);
        for_each_block(kv_begin, kv_end, conf_.mm2_k_block,
                [&](dim_t kv0, bool k_tail) {
            v_op.load_tile(kv0, 0, k_tail);
            const bool use_beta0_kernel
                    = conf_.mm2_beta != 0.0f && kv0 == kv_begin;
            assert(!(use_beta0_kernel && k_tail));
            for_each_block(
                    0, st_.p.hs_v, conf_.v_block, [&](dim_t v0, bool v_tail) {
                const brgemm_kernel_t *kernel = this->kernel(use_beta0_kernel
                                ? brg.mm2_beta0[qi_][v_tail]
                                : brg.mm2[qi_][k_tail][v_tail]);
                for (dim_t h = 0; h < brgemm_calls_; ++h) {
                    brgemm_batch_element_t be;
                    be.ptr.A = prob.of(h) + kv0 * st_.mm_dt_sz;
                    be.ptr.B = v_op.at(kv0, v0);
                    brgemm_kernel_execute(
                            kernel, 1, &be, c.f32(h) + v0, amx_wsp_);
                }
            });
        });
    }

    // Writes the item's f32 mm2 result tiles to the user output.
    void store(const item_tiles_t &pv) const {
        const auto &p = st_.p;
        const dim_t o_col_stride = p.o_strides[sdpa_pd_t::ndims - 1];
        for (dim_t h = 0; h < brgemm_calls_; ++h)
            store_pv_tile(pv.f32(h), conf_.ld_pv, heads_[h].out, conf_.out_ld,
                    o_col_stride, p.out_dt, call_m_, p.hs_v);
    }

private:
    // Returns the kernel of descriptor brg_idx, configuring AMX tiles if it
    // needs a different palette than the previous call.
    const brgemm_kernel_t *kernel(int brg_idx) {
        assert(brg_idx >= 0);
        const bool is_amx = (*conf_.brgemm.descs)[brg_idx]->is_tmm;
        st_.kernels.brgemm_palettes.maybe_tile_configure(
                is_amx, prev_brg_idx_, brg_idx);
        used_amx_ = used_amx_ || is_amx;
        const brgemm_kernel_t *kernel = st_.kernels.brgemm_kernels[brg_idx];
        assert(kernel);
        return kernel;
    }

    struct head_ptrs_t {
        const char *q;
        char *out;
        const uint8_t *cond;
    };

    const exec_state_t &st_;
    const sdpa_full_softmax_conf_t &conf_;
    const dim_t batch_idx_, kv_head_idx_;
    // Query rows of each head, and whether they form the query tail block.
    const dim_t m_;
    const int qi_;
    // Rows of each BRGEMM call, and calls per mm1/mm2 block.
    const dim_t call_m_, brgemm_calls_;
    const bool has_mm1_postops_;
    char *const thread_scratch_;
    void *const amx_wsp_;
    std::vector<head_ptrs_t> heads_;
    std::vector<const void *> rhs_;
    size_t rhs_per_head_ = 0;
    int prev_brg_idx_ = -1;
    bool used_amx_ = false;
};

// per_head and kv_head_group: each work item computes all keys of its query
// block in the thread's score, P and pv tiles.
void execute_fused(const exec_state_t &st, int nthr) {
    const auto &conf = st.conf;
    const auto &p = st.p;
    const auto &ts = conf.thread_scratch;
    const size_t q_block = static_cast<size_t>(conf.q_block);
    const dim_t n_qblk = utils::div_up(p.seq_q, conf.q_block);
    parallel_nd_ext(nthr, p.batch, p.num_head_q / conf.query_heads_per_item,
            n_qblk,
            [&](int tid, int, dim_t batch_idx, dim_t item_head_idx,
                    dim_t q_block_idx) {
        char *my_scratch = st.thread_scratch(tid);
        const item_tiles_t scores
                = {my_scratch, q_block * conf.ld_scores * sizeof(float)};
        const item_tiles_t prob = ts.has_prob
                ? item_tiles_t {my_scratch + ts.prob,
                          q_block * conf.ld_prob * st.mm_dt_sz}
                : scores;

        work_item_t item(st, tid, batch_idx, item_head_idx, q_block_idx);
        item.mm1(0, p.seq_kv, scores);
        item.softmax(scores, prob);
        if (conf.mm2_direct) {
            item.mm2(0, p.seq_kv, prob, item.output_tiles());
        } else {
            const item_tiles_t pv = {
                    my_scratch + ts.pv, q_block * conf.ld_pv * sizeof(float)};
            item.mm2(0, p.seq_kv, prob, pv);
            item.store(pv);
        }
    });
}

// split_kv: mm1, softmax and mm2 run as separate parallel phases on shared
// buffers with one row per (batch, query head, query):
//  1. mm1 tasks (work item, mm1 key chunk) write their score columns.
//  2. Softmax tasks (batch, query head, query block) turn complete score rows
//     into P rows.
//  3. mm2 tasks (work item, mm2 key chunk) write the f32 result of their chunk.
//  4. Output tasks (batch, query head, query block) sum the chunk results and
//     write the output. With a single mm2 chunk, the mm2 tasks write the
//     output instead.
// P is normalized over all keys, so the chunk results are summed as is.
void execute_split_kv(const exec_state_t &st, int nthr) {
    const auto &conf = st.conf;
    const auto &p = st.p;
    const auto &sk = conf.split_kv;
    const dim_t n_items = p.num_head_q / conf.query_heads_per_item;
    const dim_t n_qblk = utils::div_up(p.seq_q, conf.q_block);

    const size_t score_row_bytes = conf.ld_scores * sizeof(float);
    const size_t prob_row_bytes = conf.thread_scratch.has_prob
            ? conf.ld_prob * st.mm_dt_sz
            : score_row_bytes;
    const size_t pv_row_bytes = conf.ld_pv * sizeof(float);
    const size_t rows = static_cast<size_t>(p.batch) * p.num_head_q * p.seq_q;
    const size_t partial_bytes = rows * pv_row_bytes;

    auto row_index = [&](dim_t batch_idx, dim_t head, dim_t q0) {
        return static_cast<size_t>(
                (batch_idx * p.num_head_q + head) * p.seq_q + q0);
    };
    // Rows of query head `head` from query q0 in a shared buffer.
    auto shared_rows = [&](size_t buffer, size_t row_bytes, dim_t batch_idx,
                               dim_t head, dim_t q0) {
        return st.shared_area() + buffer
                + row_index(batch_idx, head, q0) * row_bytes;
    };
    // Tiles of a work item in a shared buffer.
    auto item_tiles = [&](size_t buffer, size_t row_bytes, dim_t batch_idx,
                              dim_t item_head_idx, dim_t q_block_idx) {
        return item_tiles_t {shared_rows(buffer, row_bytes, batch_idx,
                                     item_head_idx * conf.query_heads_per_item,
                                     q_block_idx * conf.q_block),
                p.seq_q * row_bytes};
    };
    // Key range [first, second) of chunk c.
    auto key_range = [&](dim_t c, dim_t chunk, dim_t n_chunks) {
        const dim_t begin = c * chunk;
        return std::make_pair(
                begin, c + 1 == n_chunks ? p.seq_kv : begin + chunk);
    };

    parallel_nd_ext(nthr, p.batch, n_items, n_qblk, sk.mm1_chunks,
            [&](int tid, int, dim_t batch_idx, dim_t item_head_idx,
                    dim_t q_block_idx, dim_t c) {
        work_item_t item(st, tid, batch_idx, item_head_idx, q_block_idx);
        const auto kv = key_range(c, sk.mm1_chunk, sk.mm1_chunks);
        item.mm1(kv.first, kv.second,
                item_tiles(sk.scores, score_row_bytes, batch_idx, item_head_idx,
                        q_block_idx));
    });

    parallel_nd(p.batch, p.num_head_q, n_qblk,
            [&](dim_t batch_idx, dim_t head, dim_t q_block_idx) {
        const dim_t q0 = q_block_idx * conf.q_block;
        softmax_rows(st,
                reinterpret_cast<float *>(shared_rows(
                        sk.scores, score_row_bytes, batch_idx, head, q0)),
                shared_rows(sk.prob, prob_row_bytes, batch_idx, head, q0),
                st.cond(batch_idx, head, q0),
                nstl::min(conf.q_block, p.seq_q - q0));
    });

    parallel_nd_ext(nthr, p.batch, n_items, n_qblk, sk.mm2_chunks,
            [&](int tid, int, dim_t batch_idx, dim_t item_head_idx,
                    dim_t q_block_idx, dim_t c) {
        work_item_t item(st, tid, batch_idx, item_head_idx, q_block_idx);
        const auto kv = key_range(c, sk.mm2_chunk, sk.mm2_chunks);
        const auto pv = item_tiles(sk.partials + c * partial_bytes,
                pv_row_bytes, batch_idx, item_head_idx, q_block_idx);
        item.mm2(kv.first, kv.second,
                item_tiles(sk.prob, prob_row_bytes, batch_idx, item_head_idx,
                        q_block_idx),
                pv);
        if (sk.mm2_chunks == 1) item.store(pv);
    });
    if (sk.mm2_chunks == 1) return;

    // Sum the chunk results into the rows of chunk 0, then write them.
    const dim_t o_row_stride = p.o_strides[sdpa_pd_t::ndims - 2];
    const dim_t o_col_stride = p.o_strides[sdpa_pd_t::ndims - 1];
    const size_t partial_floats = partial_bytes / sizeof(float);
    parallel_nd(p.batch, p.num_head_q, n_qblk,
            [&](dim_t batch_idx, dim_t head, dim_t q_block_idx) {
        const dim_t q0 = q_block_idx * conf.q_block;
        const dim_t m = nstl::min(conf.q_block, p.seq_q - q0);
        float *acc = reinterpret_cast<float *>(
                shared_rows(sk.partials, pv_row_bytes, batch_idx, head, q0));
        for (dim_t c = 1; c < sk.mm2_chunks; ++c) {
            const float *part = acc + c * partial_floats;
            for (dim_t i = 0; i < m; ++i)
                for (dim_t d = 0; d < p.hs_v; ++d)
                    acc[i * conf.ld_pv + d] += part[i * conf.ld_pv + d];
        }
        store_pv_tile(acc, conf.ld_pv,
                get_arg_ptr(st.o_td, batch_idx, head, q0), o_row_stride,
                o_col_stride, p.out_dt, m, p.hs_v);
    });
}

} // namespace

status_t execute(const sdpa_full_softmax_conf_t &conf,
        const sdpa_full_softmax_kernels_t &kernels,
        const sdpa_full_softmax_run_args_t &args, int nthr) {
    const exec_state_t st(conf, kernels, args, nthr);
    pack_ahead(st);
    if (conf.schedule == sdpa_full_softmax_schedule_t::split_kv)
        execute_split_kv(st, nthr);
    else
        execute_fused(st, nthr);
    return status::success;
}

} // namespace sdpa_full_softmax

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
