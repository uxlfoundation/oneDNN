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
#include <utility>

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
#include "common/verbose.hpp"

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

// Data type of the mm1/mm2 BRGEMM inputs (f32/f16/bf16): Q, K, V and P. The
// scores and pv tiles are always accumulated in f32 (BRGEMM C is f32); the
// packed K/V buffers and P (mm2's A operand) are materialised in this type,
// and the softmax runs on the f32 scores. The pd requires Q, K, V and dst to
// share one data type, so it is read from Q.
inline data_type_t mm_data_type(const sdpa_pd_t &pd) {
    return pd.desc()->qry_md()->data_type;
}

// Key range [first, second) of split_kv chunk c out of n_chunks. The keys are
// split evenly and each boundary is rounded down to a multiple of block; the
// last chunk ends at seq_kv. With n_chunks <= seq_kv / block, the evenly split
// boundaries are at least block apart, so every chunk starts with a full block.
std::pair<dim_t, dim_t> split_kv_chunk_keys(
        dim_t c, dim_t n_chunks, dim_t block, dim_t seq_kv) {
    auto boundary = [&](dim_t i) {
        return utils::rnd_dn(i * seq_kv / n_chunks, block);
    };
    return {boundary(c), c + 1 == n_chunks ? seq_kv : boundary(c + 1)};
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

// A B operand (K or V) of the mm1/mm2 BRGEMM: the user buffer with its memory
// descriptor and, when the tensor is packed ahead, its copy descriptor and
// scratch pointer.
struct b_tensor_desc_t {
    b_tensor_desc_t(const char *buffer, const memory_desc_t *md,
            const utils::optional_t<tensor_copy_desc_t> &copy_desc,
            char *scratch_ptr, size_t shared_area_offset)
        : buffer(buffer)
        , md(md)
        , copy_desc(copy_desc ? &*copy_desc : nullptr)
        , scratch_ptr(copy_desc ? scratch_ptr + shared_area_offset
                                  + copy_desc->scratch_offset
                                : nullptr) {}

    const char *buffer;
    memory_desc_wrapper md;
    const tensor_copy_desc_t *copy_desc;
    char *scratch_ptr;
};

// Get the pointer into a user buffer laid out as mdw for the given batch,
// head and row indices.
template <typename T>
T *get_arg_ptr(T *buffer, const memory_desc_wrapper &mdw, dim_t batch_idx,
        dim_t head_idx, dim_t row_idx = 0) {
    dim_t offset = 0;
    if (mdw.dims()[0] != 1) offset += batch_idx * mdw.strides()[0];
    if (mdw.dims()[1] != 1) offset += head_idx * mdw.strides()[1];
    if (mdw.dims()[2] != 1) offset += row_idx * mdw.strides()[2];
    return buffer + offset * types::data_type_size(mdw.data_type());
}

// Get the pointer to the copy buffer for the given batch and head indices.
char *get_copy_ptr(const b_tensor_desc_t &td, dim_t batch_idx, dim_t head_idx) {
    const auto *copy_desc = td.copy_desc;
    assert(copy_desc);
    return td.scratch_ptr + batch_idx * copy_desc->batch_stride
            + head_idx * copy_desc->head_stride;
}

// Get the pointer to the tile buffer for the given batch and head indices
// depending on whether a copy buffer is used.
const char *get_tile_ptr(
        const b_tensor_desc_t &td, dim_t batch_idx, dim_t head_idx) {
    if (td.copy_desc) return get_copy_ptr(td, batch_idx, head_idx);
    return get_arg_ptr(td.buffer, td.md, batch_idx, head_idx);
}

// Return the B-matrix offset in elements for logical coordinates [k0, n0].
size_t get_tile_offset_in_elements(
        const b_tensor_desc_t &td, dim_t k0, dim_t n0) {
    if (!td.copy_desc) {
        constexpr int row_dim = sdpa_pd_t::ndims - 2;
        constexpr int col_dim = sdpa_pd_t::ndims - 1;
        return static_cast<size_t>(k0) * td.md.strides()[row_dim]
                + static_cast<size_t>(n0) * td.md.strides()[col_dim];
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

    b_operand_t(const b_tensor_desc_t &td, dim_t batch_idx, dim_t kv_head_idx,
            const std::unique_ptr<packer_t> (&packers)[2],
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
    const b_tensor_desc_t &td_;
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
        const std::vector<sdpa_mm1_post_op_t> &mm1_post_ops,
        const sdpa_full_softmax_run_args_t &args, dim_t batch_idx,
        dim_t query_head_idx, dim_t q0, bool select_in_mm1,
        const uint8_t *cond) {
    for (size_t pi = 0; pi < mm1_post_ops.size(); ++pi) {
        const auto &pop = mm1_post_ops[pi];
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
// TODO: Implement optimized handling for strided output columns or skip
// support of non-dense columns.
void store_pv_tile(const float *pv, dim_t ld_pv, char *out,
        dim_t out_row_stride, dim_t out_col_stride, data_type_t out_dt, dim_t m,
        dim_t n) {
    const size_t o_dt_sz = types::data_type_size(out_dt);
    const bool dense_col = out_col_stride == 1;
    for (dim_t i = 0; i < m; ++i) {
        const float *prow = pv + i * ld_pv;
        char *out_row = out + i * out_row_stride * o_dt_sz;
        if (out_dt == data_type::f32) {
            float *dst = reinterpret_cast<float *>(out_row);
            if (dense_col)
                std::memcpy(dst, prow, static_cast<size_t>(n) * sizeof(float));
            else
                for (dim_t d = 0; d < n; ++d)
                    dst[d * out_col_stride] = prow[d];
        } else if (out_dt == data_type::f16) {
            float16_t *dst = reinterpret_cast<float16_t *>(out_row);
            if (dense_col)
                cvt_float_to_float16(dst, prow, static_cast<size_t>(n));
            else
                for (dim_t d = 0; d < n; ++d)
                    dst[d * out_col_stride] = float16_t(prow[d]);
        } else if (out_dt == data_type::bf16) {
            bfloat16_t *dst = reinterpret_cast<bfloat16_t *>(out_row);
            if (dense_col)
                cvt_float_to_bfloat16(dst, prow, static_cast<size_t>(n));
            else
                for (dim_t d = 0; d < n; ++d)
                    dst[d * out_col_stride] = bfloat16_t(prow[d]);
        }
    }
}

// acc[m x n] += src[m x n]; both have row stride ld. Padding columns between
// rows are not read.
void add_f32_tile(cpu_accumulator_1d_t<data_type::f32> &acc_ker, float *acc,
        const float *src, dim_t m, dim_t n, dim_t ld) {
    if (ld == n) {
        acc_ker.accumulate(acc, src, static_cast<size_t>(m) * n);
        return;
    }
    for (dim_t i = 0; i < m; ++i)
        acc_ker.accumulate(acc + i * ld, src + i * ld, static_cast<size_t>(n));
}

// Adds a partial while converting to a dense bf16/f16 output.
void add_tile_and_cvt_to_xf16(const float *acc, const float *src, dim_t ld,
        char *dst, dim_t dst_row_stride, data_type_t dst_dt, dim_t m, dim_t n) {
    const size_t dst_dt_sz = types::data_type_size(dst_dt);
    for (dim_t i = 0; i < m; ++i) {
        const float *acc_row = acc + i * ld;
        const float *src_row = src + i * ld;
        char *dst_row = dst + i * dst_row_stride * dst_dt_sz;
        if (dst_dt == data_type::bf16)
            add_floats_and_cvt_to_bfloat16(
                    reinterpret_cast<bfloat16_t *>(dst_row), acc_row, src_row,
                    n);
        else
            add_floats_and_cvt_to_float16(
                    reinterpret_cast<float16_t *>(dst_row), acc_row, src_row,
                    n);
    }
}

status_t set_mm1_post_ops(brgemm_desc_t &brg, dim_t output_rows,
        dim_t full_output_width, dim_t output_row_stride,
        const std::vector<sdpa_mm1_post_op_t> *post_ops, bool select_as_postop,
        bool stack_heads) {
    const bool has_post_ops = post_ops && !post_ops->empty();
    if (!has_post_ops && !select_as_postop) return status::success;

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
                const dim_t rows
                        = pop.rhs_dims[row_axis] == 1 ? 1 : output_rows;

                constexpr int key_axis = sdpa_pd_t::mask_k_index;
                const dim_t cols
                        = pop.rhs_dims[key_axis] == 1 ? 1 : full_output_width;

                dims_t rhs_dims = {rows, cols};
                dims_t rhs_str = {
                        pop.rhs_strides[row_axis], pop.rhs_strides[key_axis]};

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
        dims_t cd_dims = {output_rows, full_output_width};
        CHECK(memory_desc_init_by_tag(
                cond_md, 2, cd_dims, data_type::u8, format_tag::ab));

        CHECK(po.append_binary(alg_kind::binary_select, &fill_md, &cond_md));
    }

    primitive_attr_t attr;
    CHECK(attr.set_post_ops(po));

    memory_desc_t dst_md;
    dims_t d_dims = {output_rows, full_output_width};
    dims_t d_strides = {output_row_stride, 1};
    CHECK(memory_desc_init_by_strides(
            dst_md, 2, d_dims, data_type::f32, d_strides));
    return brgemm_desc_set_postops(&brg, &attr, &dst_md, output_row_stride);
}

status_t init_brgemm_desc(brgemm_desc_t &brg, data_type_t dt, float beta,
        dim_t M, dim_t N, dim_t K, dim_t lda, dim_t ldb, dim_t ldc) {
    return brgemm_desc_init(&brg, isa_undef, brgemm_addr, dt, dt,
            /*transA=*/false, /*transB=*/false, brgemm_row_major,
            /*alpha=*/1.0f, beta, lda, ldb, ldc, M, N, K, /*strides=*/nullptr);
}

status_t finalize_brgemm_desc(brgemm_desc_t &brg, bool b_is_vnni) {
    const bool amx_vnni_k_tail = (brg.is_bf16_tmm || brg.is_f16_tmm)
            && brg.reduce_dim % data_type_vnni_granularity(brg.dt_a) != 0;
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

    return brgemm_desc_finalize(&brg);
}

status_t build_mm1_brgemm_desc(brgemm_desc_t &brg, const sdpa_pd_t &pd,
        const sdpa_full_softmax_conf_t &conf, dim_t M, dim_t N) {
    const sdpa_desc_t &desc = *pd.desc();
    CHECK(init_brgemm_desc(brg, mm_data_type(pd), /*beta=*/0.0f, M, N,
            desc.head_size(), conf.mm1_lda, conf.ldk, conf.ld_scores));

    // N is the kv-block width, but post-op descriptors cover the full
    // [M, seq_kv] score tile. At execution, data_C_ptr_ anchors that full tile
    // while ptr_D points at the current N-block; the binary injector uses
    // their difference to recover each element's row and global column.
    // This handles per-element and broadcast masks.
    CHECK(set_mm1_post_ops(brg, M, desc.keys(), conf.ld_scores,
            &conf.mm1_post_ops, conf.mm1_select_postop, conf.stack_heads));

    return finalize_brgemm_desc(brg, conf.mm1_needs_vnni_b);
}

status_t build_mm2_brgemm_desc(brgemm_desc_t &brg, const sdpa_pd_t &pd,
        const sdpa_full_softmax_conf_t &conf, float beta, dim_t M, dim_t N,
        dim_t K) {
    CHECK(init_brgemm_desc(brg, mm_data_type(pd), beta, M, N, K, conf.ld_prob,
            conf.ldv, conf.mm2_ldc));

    if (conf.mm2_postops) {
        memory_desc_t dst_md;
        dims_t dst_dims = {M, N};
        dims_t dst_strides = {conf.out_ld, 1};
        CHECK(memory_desc_init_by_strides(
                dst_md, 2, dst_dims, pd.dst_md()->data_type, dst_strides));
        CHECK(brgemm_desc_set_postops(&brg, nullptr, &dst_md, conf.out_ld));
    }

    return finalize_brgemm_desc(brg, conf.mm2_needs_vnni_b);
}

status_t insert_brgemm_desc(brgemm_containers::brgemm_desc_container_t &descs,
        brgemm_desc_t &brg, bool needs_vnni, int &idx, size_t &max_wsp) {
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
    idx = descs.insert(brg);
    return status::success;
}

status_t build_brgemm_descs(const sdpa_pd_t &pd,
        const sdpa_full_softmax_conf_t &conf,
        sdpa_full_softmax_brgemm_t &brg_conf, size_t &max_wsp) {
    brg_conf = {};
    // The descriptor container is shared across pd clones.
    brg_conf.descs
            = std::make_shared<brgemm_containers::brgemm_desc_container_t>();
    auto &descs = *brg_conf.descs;

    const dim_t ms[2] = {conf.m_block, conf.q_tail};
    const dim_t ns[2] = {conf.mm1_n_block, conf.mm1_n_tail};
    for (int qi = 0; qi < 2; ++qi) {
        if (!ms[qi]) continue;
        for (int ni = 0; ni < 2; ++ni) {
            if (!ns[ni]) continue;
            brgemm_desc_t brg;
            CHECK(build_mm1_brgemm_desc(brg, pd, conf, ms[qi], ns[ni]));
            CHECK(insert_brgemm_desc(descs, brg, conf.mm1_needs_vnni_b,
                    brg_conf.mm1[qi][ni], max_wsp));
        }
    }

    // Build the configured mm2 variants and, for a tiled reduction, beta-zero
    // variants for the first K block.
    const dim_t ks[2] = {conf.mm2_k_block, conf.mm2_k_tail};
    for (int qi = 0; qi < 2; ++qi) {
        if (!ms[qi]) continue;
        for (int ki = 0; ki < 2; ++ki) {
            if (!ks[ki]) continue;
            for (int vi = 0; vi < 2; ++vi) {
                const dim_t width = vi ? conf.v_tail : conf.v_block;
                if (!width) continue;
                brgemm_desc_t brg;
                CHECK(build_mm2_brgemm_desc(
                        brg, pd, conf, conf.mm2_beta, ms[qi], width, ks[ki]));
                CHECK(insert_brgemm_desc(descs, brg, conf.mm2_needs_vnni_b,
                        brg_conf.mm2[qi][ki][vi], max_wsp));
            }
        }
        if (conf.mm2_beta != 0.0f) {
            for (int vi = 0; vi < 2; ++vi) {
                const dim_t width = vi ? conf.v_tail : conf.v_block;
                if (!width) continue;
                brgemm_desc_t brg;
                CHECK(build_mm2_brgemm_desc(brg, pd, conf, /*beta=*/0.0f,
                        ms[qi], width, conf.mm2_k_block));
                CHECK(insert_brgemm_desc(descs, brg, conf.mm2_needs_vnni_b,
                        brg_conf.mm2_beta0[qi][vi], max_wsp));
            }
        }
    }
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

tensor_copy_desc_t make_b_copy_desc(size_t scratch_offset, dim_t rows,
        dim_t cols, data_type_t mm_dt, dim_t num_head_kv, bool needs_vnni) {
    using b_packer_t = sdpa_brgemm_copy::b_packer_t;
    tensor_copy_desc_t copy_desc;
    copy_desc.scratch_offset = scratch_offset;
    copy_desc.padded_k = b_packer_t::packed_k(rows, mm_dt);
    copy_desc.vnni_factor = needs_vnni ? data_type_vnni_granularity(mm_dt) : 1;
    copy_desc.head_stride = static_cast<size_t>(copy_desc.padded_k)
            * b_packer_t::packed_ld(cols) * types::data_type_size(mm_dt);
    copy_desc.batch_stride
            = static_cast<size_t>(num_head_kv) * copy_desc.head_stride;
    return copy_desc;
}

size_t add_shared_buffer(size_t bytes, size_t &shared_offset) {
    const size_t offset = align64(shared_offset);
    shared_offset = offset + bytes;
    return offset;
}

void layout_scratch(const sdpa_desc_t &desc, data_type_t mm_dt, size_t max_wsp,
        sdpa_full_softmax_conf_t &conf) {
    const bool split_kv
            = conf.schedule == sdpa_full_softmax_schedule_t::split_kv;
    auto &ts = conf.thread_scratch;
    // Fused schedules keep score, P and pv tiles for each head of the work
    // item in its thread's scratch. split_kv keeps these tiles in shared
    // buffers instead.
    const size_t item_q_rows = split_kv
            ? 0
            : static_cast<size_t>(conf.query_heads_per_item) * conf.q_block;
    conf.amx_wsp_bytes = align64(max_wsp);
    ts.set_tile_buffers(item_q_rows, conf.ld_scores, conf.ld_prob, conf.ld_pv,
            conf.mm2_direct, conf.amx_wsp_bytes);

    const dim_t batch = desc.batch();
    const dim_t num_head_kv = desc.num_kv_heads();
    const dim_t seq_kv = desc.keys();
    const dim_t hs_qk = desc.head_size();
    const dim_t hs_v = desc.values();
    size_t shared_offset = 0;
    if (conf.kv_packing == sdpa_full_softmax_kv_packing_t::per_tile) {
        // A K tile is one mm1 N block (at most one panel); a V tile is one mm2
        // K block across all of hs_v.
        if (conf.pack_k) {
            conf.k_tile_desc
                    = make_b_copy_desc(ts.size, hs_qk, conf.mm1_n_block, mm_dt,
                            num_head_kv, conf.mm1_needs_vnni_b);
            ts.size += align64(conf.k_tile_desc->head_stride);
        }
        if (conf.pack_v) {
            conf.v_tile_desc = make_b_copy_desc(ts.size, conf.mm2_k_block, hs_v,
                    mm_dt, num_head_kv, conf.mm2_needs_vnni_b);
            ts.size += align64(conf.v_tile_desc->head_stride);
        }
    } else {
        // Whole packed tensors are placed consecutively in shared scratch.
        if (conf.pack_k) {
            conf.k_copy_desc = make_b_copy_desc(shared_offset, hs_qk, seq_kv,
                    mm_dt, num_head_kv, conf.mm1_needs_vnni_b);
            shared_offset += static_cast<size_t>(batch)
                    * conf.k_copy_desc->batch_stride;
        }
        if (conf.pack_v) {
            conf.v_copy_desc = make_b_copy_desc(shared_offset, seq_kv, hs_v,
                    mm_dt, num_head_kv, conf.mm2_needs_vnni_b);
            shared_offset += static_cast<size_t>(batch)
                    * conf.v_copy_desc->batch_stride;
        }
    }
    if (split_kv) {
        auto &sk = conf.split_kv;
        const size_t rows = static_cast<size_t>(batch) * desc.num_q_heads()
                * desc.queries();
        sk.scores = add_shared_buffer(
                rows * conf.ld_scores * sizeof(float), shared_offset);
        sk.prob = ts.has_prob ? add_shared_buffer(rows * conf.ld_prob
                                                * types::data_type_size(mm_dt),
                                        shared_offset)
                              : sk.scores;
        if (!conf.mm2_direct)
            sk.partials = add_shared_buffer(
                    sk.mm2_chunks * rows * conf.ld_pv * sizeof(float),
                    shared_offset);
    }
    conf.scratch_per_thread = ts.size;
    conf.shared_scratch_bytes = shared_offset;
}

} // namespace

sdpa_full_softmax_kernels_t::sdpa_full_softmax_kernels_t() = default;
sdpa_full_softmax_kernels_t::~sdpa_full_softmax_kernels_t() = default;

namespace sdpa_full_softmax {

status_t configure(const sdpa_pd_t &pd, const sdpa_full_softmax_knobs_t &knobs,
        sdpa_full_softmax_conf_t &conf) {
    conf = sdpa_full_softmax_conf_t();

    constexpr int ndims = sdpa_pd_t::ndims;
    constexpr int row_dim = ndims - 2;

    const sdpa_desc_t &desc = *pd.desc();
    const memory_desc_wrapper q_mdw(desc.qry_md()), k_mdw(desc.key_md()),
            v_mdw(desc.val_md()), out_mdw(pd.dst_md()),
            cond_mdw(desc.attn_mask_md());
    const dim_t batch = desc.batch();
    const dim_t num_head_q = desc.num_q_heads();
    const dim_t num_head_kv = desc.num_kv_heads();
    // GQA group size (>= 1).
    const dim_t group_head = num_head_q / num_head_kv;
    const dim_t seq_q = desc.queries();
    const dim_t seq_kv = desc.keys();
    const dim_t hs_qk = desc.head_size();
    const dim_t hs_v = desc.values();
    const data_type_t mm_dt = mm_data_type(pd);
    // SDPA output data type; f32 accumulators are converted to it.
    const data_type_t out_dt = out_mdw.data_type();
    const dims_t &q_strides = q_mdw.strides();
    const dims_t &k_strides = k_mdw.strides();
    const dims_t &v_strides = v_mdw.strides();
    const dims_t &out_strides = out_mdw.strides();
    const bool has_select = pd.with_select_mask();
    // When true, keep scores where cond == 0; otherwise keep where cond != 0.
    const bool invert_select = desc.invert_select;
    // Null without a select mask.
    const dim_t *cond_dims = has_select ? cond_mdw.dims() : nullptr;
    const dim_t *cond_strides = has_select ? cond_mdw.strides() : nullptr;

    if (!utils::one_of(mm_dt, data_type::f32, data_type::f16, data_type::bf16))
        return status::unimplemented;

    if (!utils::one_of(out_dt, data_type::f32, data_type::f16, data_type::bf16))
        return status::unimplemented;

    // mm1 post-ops in chain order: the scale (a scalar, read as f32 at
    // execution) and then the additive attention mask. Select is handled
    // separately.
    if (pd.with_attn_scale()) {
        sdpa_mm1_post_op_t sc;
        sc.alg = alg_kind::binary_mul;
        sc.is_binary = true;
        sc.rhs_is_scalar = true;
        sc.rhs_dt = data_type::f32;
        conf.mm1_post_ops.push_back(sc);
    }
    if (pd.with_buffer_mask()) {
        const memory_desc_wrapper mask_mdw(desc.attn_mask_md());
        sdpa_mm1_post_op_t mk;
        mk.alg = alg_kind::binary_add;
        mk.is_binary = true;
        mk.rhs_is_scalar = false;
        mk.rhs_dt = mask_mdw.data_type();
        mk.rhs_dims.assign(mask_mdw.dims(), mask_mdw.dims() + ndims);
        mk.rhs_strides.assign(mask_mdw.strides(), mask_mdw.strides() + ndims);
        conf.mm1_post_ops.push_back(mk);
    }

    const dim_t mm_dt_sz = static_cast<dim_t>(types::data_type_size(mm_dt));

    const dim_t k_seq_stride = k_strides[ndims - 1];
    const dim_t k_hs_stride = k_strides[row_dim];

    const bool mm1_transpose_k = k_seq_stride != 1;
    const bool transpose_v = v_strides[ndims - 1] != 1;

    CHECK(is_vnni_b_required(mm_dt, mm1_transpose_k, conf.mm1_needs_vnni_b));
    CHECK(is_vnni_b_required(mm_dt, transpose_v, conf.mm2_needs_vnni_b));

    const bool need_copy_v = conf.mm2_needs_vnni_b || transpose_v;

    // Width of one packed K/V panel. A packed B operand is read by BRGEMM with
    // ldb = b_panel_n and an N block of at most b_panel_n (see b_packer_t).
    constexpr dim_t b_panel_n = sdpa_brgemm_copy::b_packer_t::panel_n;

    const dim_t l2_budget_bytes
            = static_cast<dim_t>(3 * platform::get_per_core_cache_size(2) / 4);

    bool is_amx = false;
    CHECK(uses_amx_tiles(mm_dt, is_amx));
    conf.nthr = dnnl_get_max_threads();

    // Fold non-inverted select into mm1 when condition rows match dense score
    // rows. The pd guarantees a full, contiguous key axis. A single query row
    // does not use its row stride; otherwise require non-broadcast query rows
    // with stride seq_kv.
    conf.mm1_select_postop = false;
    if (has_select) {
        const bool dense_query_axis = seq_q == 1
                || (cond_dims[sdpa_pd_t::mask_q_index] != 1
                        && cond_strides[sdpa_pd_t::mask_q_index] == seq_kv);
        conf.mm1_select_postop = !invert_select && dense_query_axis;
    }

    // The binary injector requires tensor rhs data to share the score tile's
    // layout. Keep score rows dense when a tensor rhs is used; scalar rhs and
    // eltwise post-ops are layout-independent.
    bool dense_scores_required = conf.mm1_select_postop;
    for (const auto &pop : conf.mm1_post_ops)
        if (pop.is_binary && !pop.rhs_is_scalar) dense_scores_required = true;
    conf.ld_scores
            = scratch_row_ld(seq_kv, sizeof(float), !dense_scores_required);
    auto &thread_scratch = conf.thread_scratch;
    thread_scratch.set_compute_type(mm_dt, mm_dt_sz);
    conf.ld_prob = thread_scratch.has_prob
            ? scratch_row_ld(seq_kv, mm_dt_sz, /*allow_padding=*/true)
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
    const float head_scale = std::sqrt((hs_qk + hs_v) / 128.f);
    const float score_tile_target_bytes = score_tile_l2_ratio * head_scale
            * platform::get_per_core_cache_size(2);
    const float q_rows
            = score_tile_target_bytes / static_cast<float>(score_row_bytes);
    // A long seq_kv shrinks the tile-based q_block until K/V streaming
    // dominates. Bound it so the K/V bytes per score element,
    // (hs_qk + hs_v) * mm_dt_sz / q_block, stay below the ~16..24 bytes of
    // score-tile traffic per element. Threads working on the same KV head at
    // the same time share its K/V through L3, which halves the bound once more
    // than two threads do. That is the case when there are more threads than
    // twice the KV heads, or with interleave_items: per_head and
    // kv_head_group then deal work items to threads round-robin, so
    // concurrent threads work on neighboring query blocks of one KV head.
    // With a contiguous range of work items per thread, each thread instead
    // works on its own KV head and reuses its K/V from L2 across its query
    // blocks. That is faster while the K/V of all concurrently processed KV
    // heads stay in L3 next to the threads' score tiles, but once they
    // overflow, every query block reads K/V from memory again (GNR, bf16,
    // 8x8x16385x256 GQA: 3.2 s contiguous vs 1.9 s interleaved; 4x32x1024x128,
    // which fits: 4.3 ms vs 5.4 ms). Interleaving needs several query blocks
    // per head; q_block never exceeds q_block_max.
    const dim_t kv_heads_total = batch * num_head_kv;
    const double concurrent_kv_bytes
            = static_cast<double>(nstl::min<dim_t>(conf.nthr, kv_heads_total))
            * seq_kv * (hs_qk + hs_v) * mm_dt_sz;
    auto kv_overflows_l3 = [&](double score_tile_bytes) {
        const double l3_left_bytes = static_cast<double>(conf.nthr)
                * (static_cast<double>(platform::get_per_core_cache_size(3))
                        - score_tile_bytes);
        return seq_q > q_block_max && concurrent_kv_bytes > l3_left_bytes;
    };
    // Estimate L3 occupancy before choosing the schedule and q_block; this
    // decision affects q_rows_min. The L2-based score-tile target may exceed
    // the tile for q_block_max rows, so cap the estimate at that size.
    // For fused schedules, if interleaving is not already enabled, a later
    // L3 check uses the actual tile size after choosing q_block and
    // query_heads_per_item. Grouped-head rounding can make that tile larger
    // than this estimate, enabling interleaving.
    const double estimated_score_tile_bytes
            = nstl::min<double>(score_tile_target_bytes,
                    static_cast<double>(q_block_max) * score_row_bytes);
    conf.interleave_items = kv_overflows_l3(estimated_score_tile_bytes);
    const bool kv_shared_in_l3
            = 2 * kv_heads_total < conf.nthr || conf.interleave_items;
    const float q_rows_min = (hs_qk + hs_v) / (kv_shared_in_l3 ? 16.f : 8.f);
    // Query rows of one work item: per-head q_block, or the rows of all heads
    // in a KV-head group.
    dim_t item_rows = q_granule
            * std::lround(nstl::max(q_rows, q_rows_min) / q_granule);
    item_rows = utils::saturate(q_granule, q_block_max, item_rows);

    // Query block of a work item with heads_per_item query heads: its share of
    // item_rows, rounded down to q_granule, then shrunk while that leaves
    // fewer work items than threads.
    auto choose_q_block = [&](dim_t heads_per_item) {
        dim_t qb = item_rows / heads_per_item;
        if (qb < seq_q) qb = nstl::max(q_granule, utils::rnd_dn(qb, q_granule));
        qb = nstl::min(qb, seq_q);
        const dim_t head_items = batch * num_head_q / heads_per_item;
        while (qb > q_granule
                && head_items * utils::div_up(seq_q, qb) < conf.nthr)
            qb = utils::rnd_dn(qb - 1, q_granule);
        return utils::saturate((dim_t)1, seq_q, qb);
    };

    // Choose mm1's N block before estimating schedules so the split estimate
    // uses the same block-aligned chunk boundaries as execution.
    constexpr dim_t kv_granule_large = b_panel_n;
    constexpr dim_t kv_granule_medium = 32;
    constexpr dim_t kv_granule_min = 16;
    const dim_t b_panel_budget_bytes = l2_budget_bytes / 8;
    dim_t kv_block = b_panel_budget_bytes / (hs_qk * mm_dt_sz);
    const dim_t granule = kv_block >= kv_granule_large
            ? kv_granule_large
            : (kv_block >= kv_granule_medium ? kv_granule_medium
                                             : kv_granule_min);
    kv_block = nstl::max<dim_t>(
            utils::rnd_dn(kv_block, granule), kv_granule_min);
    kv_block = nstl::min<dim_t>(kv_block, seq_kv);

    const dim_t k_reuse_rows = seq_q * group_head;
    const float k_l2_set_load
            = f32_tile_l2_set_load(hs_qk, kv_block, k_hs_stride);
    const bool pack_k_for_aliasing = mm_dt == data_type::f32 && !mm1_transpose_k
            && ((k_reuse_rows >= 256 && k_l2_set_load > 0.5f)
                    || (k_reuse_rows >= 128 && k_l2_set_load >= 1.0f));
    const bool need_copy_k
            = mm1_transpose_k || conf.mm1_needs_vnni_b || pack_k_for_aliasing;
    const dim_t mm1_n_block
            = need_copy_k ? nstl::min<dim_t>(kv_block, b_panel_n) : kv_block;
    constexpr dim_t amx_mm2_k_block = 256;
    const dim_t mm2_k_block = is_amx ? amx_mm2_k_block : 512;

    // Estimated time in ns of a schedule with heads_per_item query heads per
    // work item and query block. Each phase uses the same contiguous
    // balance211 task ranges as parallel_nd_ext; split mm1 and mm2 are modeled
    // separately because each has its own block size and actual chunk count.
    // Chunk lengths use the same rounded-down boundaries as execution.
    // Forced-count leave-one-host-out validation found no reliable benefit
    // from refitting these coefficients, so keep them until a better model is
    // validated on additional hosts.
    struct phase_work_t {
        double keys = 0;
        double row_keys = 0;
        double mac_rows_keys = 0;
    };
    struct cost_coefs_t {
        double kv_byte, mac, score, total_kv_byte, merge_elem;
    };
    const cost_coefs_t coefs = is_amx
            ? cost_coefs_t {0.042, 0.00246, 1.26, 0.00152, 12.7}
            : cost_coefs_t {0.0263, 0.0308, 0.0, 0.000402, 18.1};
    auto phase_work = [&](dim_t groups, dim_t heads_per_item, dim_t qb,
                              dim_t chunks, dim_t block, dim_t key_count) {
        phase_work_t peak;
        const dim_t n_qblk = utils::div_up(seq_q, qb);
        const dim_t tasks_per_group = n_qblk * chunks;
        const dim_t work_amount = groups * tasks_per_group;
        if (work_amount == 0) return peak;

        const int nthr
                = static_cast<int>(nstl::min<dim_t>(conf.nthr, work_amount));
        const dim_t full_qblks = n_qblk - 1;
        const dim_t tail_rows = seq_q - full_qblks * qb;
        const dim_t full_rows = heads_per_item * qb;
        const dim_t tail_work_rows = heads_per_item * tail_rows;
        const dim_t full_mac_rows
                = heads_per_item * (is_amx ? utils::rnd_up(qb, q_granule) : qb);
        const dim_t tail_mac_rows = heads_per_item
                * (is_amx ? utils::rnd_up(tail_rows, q_granule) : tail_rows);
        const dim_t sum_rows = full_qblks * full_rows + tail_work_rows;
        const dim_t sum_mac_rows = full_qblks * full_mac_rows + tail_mac_rows;
        auto chunk_boundary = [&](dim_t c) {
            return c == chunks ? key_count
                               : utils::rnd_dn(c * key_count / chunks, block);
        };
        auto prefix_keys = [&](dim_t task) {
            const dim_t group_tasks = n_qblk * chunks;
            const dim_t complete_groups = task / group_tasks;
            const dim_t remainder = task % group_tasks;
            const dim_t qblk = remainder / chunks;
            const dim_t chunk = remainder % chunks;
            return static_cast<double>(complete_groups) * n_qblk * key_count
                    + qblk * key_count + chunk_boundary(chunk);
        };
        auto prefix_rows = [&](dim_t task, bool mac) {
            const dim_t group_tasks = n_qblk * chunks;
            const dim_t complete_groups = task / group_tasks;
            const dim_t remainder = task % group_tasks;
            const dim_t qblk = remainder / chunks;
            const dim_t chunk = remainder % chunks;
            const dim_t full_rows_per_qblk = mac ? full_mac_rows : full_rows;
            const dim_t tail_rows_per_qblk
                    = mac ? tail_mac_rows : tail_work_rows;
            const dim_t sum_rows_per_group = mac ? sum_mac_rows : sum_rows;
            const dim_t rows_per_qblk = mac ? full_mac_rows : full_rows;
            const dim_t full_qblk_count = nstl::min(qblk, full_qblks);
            return static_cast<double>(complete_groups) * key_count
                    * sum_rows_per_group
                    + static_cast<double>(key_count) * full_qblk_count
                    * full_rows_per_qblk
                    + static_cast<double>(chunk_boundary(chunk))
                    * (qblk == full_qblks ? tail_rows_per_qblk : rows_per_qblk);
        };

        for (int tid = 0; tid < nthr; ++tid) {
            dim_t start = 0, end = 0;
            balance211(work_amount, nthr, tid, start, end);
            const double keys = prefix_keys(end) - prefix_keys(start);
            const double row_keys
                    = prefix_rows(end, false) - prefix_rows(start, false);
            const double mac_rows_keys
                    = prefix_rows(end, true) - prefix_rows(start, true);
            peak.keys = nstl::max(peak.keys, keys);
            peak.row_keys = nstl::max(peak.row_keys, row_keys);
            peak.mac_rows_keys = nstl::max(peak.mac_rows_keys, mac_rows_keys);
        }
        return peak;
    };
    // Without stacked heads, a work item runs one BRGEMM call per query head
    // and each call reads the item's K/V block again, mostly from L1/L2 (GNR,
    // 37 bf16/f32 prefill GQA shapes: half the cost of the first read fits
    // best). Across GNR, SPR and ICX, group was slower than per_head for
    // nearly all seq_q = 2..16 GQA cases, where these calls have 2..16 rows.
    const bool heads_may_stack = seq_q == 1 && knobs.allow_stack_heads;
    auto head_call_kv_reads = [&](dim_t heads_per_item) {
        return heads_may_stack ? 1.0 : 1.0 + 0.5 * (heads_per_item - 1);
    };
    auto schedule_cost = [&](dim_t heads_per_item, dim_t qb, dim_t chunks) {
        const dim_t hs = hs_qk + hs_v;
        const double kv_bytes_per_key = static_cast<double>(hs) * mm_dt_sz;
        const dim_t groups = batch * num_head_q / heads_per_item;
        double t = 0;
        dim_t mm2_chunks = 1;
        if (chunks == 1) {
            const phase_work_t work
                    = phase_work(groups, heads_per_item, qb, 1, seq_kv, seq_kv);
            t = coefs.kv_byte * work.keys * kv_bytes_per_key
                            * head_call_kv_reads(heads_per_item)
                    + coefs.mac * work.mac_rows_keys * hs
                    + coefs.score * 2 * work.row_keys;
        } else {
            const dim_t mm1_chunks = nstl::max<dim_t>(
                    1, nstl::min(chunks, seq_kv / mm1_n_block));
            mm2_chunks = nstl::max<dim_t>(
                    1, nstl::min(chunks, seq_kv / mm2_k_block));
            const phase_work_t mm1 = phase_work(groups, heads_per_item, qb,
                    mm1_chunks, mm1_n_block, seq_kv);
            const phase_work_t mm2 = phase_work(groups, heads_per_item, qb,
                    mm2_chunks, mm2_k_block, seq_kv);
            const phase_work_t softmax
                    = phase_work(batch * num_head_q, 1, qb, 1, seq_kv, seq_kv);
            const phase_work_t merge
                    = phase_work(batch * num_head_q, 1, qb, 1, 1, 1);
            t = coefs.kv_byte * (mm1.keys * hs_qk + mm2.keys * hs_v) * mm_dt_sz
                            * head_call_kv_reads(heads_per_item)
                    + coefs.mac
                            * (mm1.mac_rows_keys * hs_qk
                                    + mm2.mac_rows_keys * hs_v)
                    + coefs.score * (mm1.row_keys + softmax.row_keys);
            if (mm2_chunks > 1)
                t += coefs.merge_elem * merge.row_keys * hs_v
                        * (mm2_chunks - 1);
        }
        const dim_t kv_reads = heads_per_item == 1 ? group_head : 1;
        t += coefs.total_kv_byte * kv_reads * kv_heads_total * seq_kv
                * kv_bytes_per_key;
        return t;
    };

    // Choose the schedule (if `any`) and the split_kv chunk count by the
    // smallest estimated time. split_kv is a candidate only when the
    // KV-head-group work items alone are fewer than threads: its shared
    // score/P buffers hold all batch x query head x seq_q rows, which stays
    // small in that case. More chunks per item than threads never help, and
    // every chunk must hold at least one full mm2 block (see below). mm1 can
    // still run fewer chunks if its block is longer than mm2's (unpacked f32
    // K with a small head).
    using schedule_t = sdpa_full_softmax_schedule_t;
    const dim_t group_q_block = choose_q_block(group_head);
    const dim_t group_items
            = kv_heads_total * utils::div_up(seq_q, group_q_block);
    dim_t split_chunks = knobs.split_kv_chunks;
    if (split_chunks <= 0) {
        split_chunks = 1;
        double t_min = schedule_cost(group_head, group_q_block, 1);
        const dim_t max_chunks
                = nstl::min<dim_t>(conf.nthr, seq_kv / mm2_k_block);
        for (dim_t c = 2; c <= max_chunks; ++c) {
            const double t = schedule_cost(group_head, group_q_block, c);
            if (t < t_min) {
                t_min = t;
                split_chunks = c;
            }
        }
    }
    conf.schedule = knobs.schedule;
    if (conf.schedule == schedule_t::any) {
        conf.schedule = schedule_t::per_head;
        double best = schedule_cost(1, choose_q_block(1), 1);
        if (group_head > 1) {
            const double t = schedule_cost(group_head, group_q_block, 1);
            if (t < best) {
                best = t;
                conf.schedule = schedule_t::kv_head_group;
            }
        }
        if (group_items < conf.nthr && split_chunks > 1
                && schedule_cost(group_head, group_q_block, split_chunks)
                        < best)
            conf.schedule = schedule_t::split_kv;
    }
    conf.query_heads_per_item
            = conf.schedule == schedule_t::per_head ? 1 : group_head;
    const bool split_kv = conf.schedule == schedule_t::split_kv;

    const dim_t q_block = choose_q_block(conf.query_heads_per_item);
    conf.q_block = q_block;
    const dim_t q_tail = seq_q % q_block;
    conf.q_tail = q_tail;

    // Repeat the L3 check with the chosen schedule's score tile. It exceeds
    // the target when q_rows_min raises q_block or when kv_head_group rounds
    // each head's q_block up to q_granule (GNR, bf16 1x32x16384x128 GQA:
    // 4 heads x 16 rows x 16384 keys x 6 B = 6.3 MB per thread versus a
    // 2.1 MB target; interleaving made it 25..31% faster). This check only
    // turns interleaving on and keeps q_block: interleaving with the larger
    // unshared q_rows_min was still 8% faster than the contiguous order
    // (bf16 8x8x16385x256).
    if (!split_kv && !conf.interleave_items)
        conf.interleave_items = kv_overflows_l3(static_cast<double>(
                conf.query_heads_per_item * q_block * score_row_bytes));

    // An f32 K already laid out as [hs_qk, seq_kv] is read in place by mm1 as
    // [hs_qk x kv_block] B panels. If the K row stride maps many panel rows to
    // the same L2 sets, the panel is evicted between re-reads, so it is packed
    // into dense b_panel_n-wide panels instead. The panel shares L2 with the
    // score tile, Q, the V panel and prefetched lines, so it starts thrashing well
    // before it alone fills a set. Packing costs one extra pass over K per KV
    // head and only pays off when the panel is re-read by enough query rows
    // (seq_q * group_head): moderate aliasing needs more reuse to amortize it
    // than a panel that alone overflows a set.
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
    conf.kv_packing = knobs.kv_packing;
    if (conf.kv_packing == kv_packing_t::any)
        conf.kv_packing = split_kv || conf.q_block == seq_q
                ? kv_packing_t::per_tile
                : kv_packing_t::ahead;

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
    // (GNR). The block depth is set before the schedule choice (mm2_k_block).
    conf.mm1_n_block = mm1_n_block;
    conf.mm1_n_tail = seq_kv % conf.mm1_n_block;
    conf.mm2_k_block = nstl::min(seq_kv, mm2_k_block);
    conf.mm2_k_tail = seq_kv % conf.mm2_k_block;
    // >1 block -> mm2 accumulates across blocks: the first block uses a beta = 0
    // kernel (fresh C) and the rest beta = 1, so no destination pre-zeroing is
    // needed. A single block keeps the original fresh-C (beta = 0) path.
    const float mm2_beta = conf.mm2_k_block < seq_kv ? 1.0f : 0.0f;
    conf.mm2_beta = mm2_beta;

    // split_kv splits each work item's seq_kv range into split_chunks chunks
    // (chosen above), but into at most as many chunks as full mm1/mm2 blocks,
    // so every chunk starts with a full block (see split_kv_chunk_keys()): the
    // block/tail descriptors stay valid, and mm2 starts every chunk with its
    // beta = 0 kernel.
    if (split_kv) {
        auto n_chunks = [&](dim_t block) {
            return nstl::max<dim_t>(1, nstl::min(split_chunks, seq_kv / block));
        };
        auto &sk = conf.split_kv;
        sk.mm1_chunks = n_chunks(conf.mm1_n_block);
        sk.mm2_chunks = n_chunks(conf.mm2_k_block);
    }

    // mm2 can write the final result directly to the output when it has only
    // one chunk, skipping the pv scratch tile and the down-convert/scatter
    // for f32 or converting in the final K block's epilogue for bf16/f16.
    // Multiple mm2 chunks must be summed before writing the output, and a
    // strided output column keeps pv+scatter.
    const dim_t out_col_stride = out_strides[ndims - 1];
    const bool mm2_has_single_chunk
            = !split_kv || conf.split_kv.mm2_chunks == 1;
    const bool mm2_direct = mm2_has_single_chunk && out_col_stride == 1
            && out_dt == data_type::f32;
    conf.mm2_direct = mm2_direct;
    conf.mm2_postops = mm2_has_single_chunk && out_col_stride == 1
            && utils::one_of(out_dt, data_type::bf16, data_type::f16);
    conf.ld_pv = scratch_row_ld(hs_v, sizeof(float), /*allow_padding=*/true);

    // With seq_q == 1, the queries in a kv_head_group work item come from
    // different heads and can be processed together as BRGEMM rows.
    // The per-head score/P/pv tiles of a work item are already consecutive
    // [1 x ld] rows. Every row-addressed operand must then use its head stride
    // as the row stride:
    //  - Q: head rows must not overlap (a broadcast Q has head stride 0);
    //  - a tensor mm1 RHS (mask) and a folded select condition: the binary
    //    injector addresses them with score-tile offsets, so they must either
    //    broadcast over heads or have a head stride equal to ld_scores.
    // Output head rows never overlap: the memory descriptor checks non-zero
    // strides, and the pd rejects zero dst strides on non-unit dimensions.
    constexpr int head_dim = 1;
    auto can_stack_heads = [&]() {
        if (!knobs.allow_stack_heads || seq_q != 1
                || conf.query_heads_per_item == 1)
            return false;
        if (q_strides[head_dim] < hs_qk) return false;
        for (const auto &pop : conf.mm1_post_ops) {
            if (!pop.is_binary || pop.rhs_is_scalar) continue;
            if (pop.rhs_dims[head_dim] != 1
                    && pop.rhs_strides[head_dim] != conf.ld_scores)
                return false;
        }
        if (conf.mm1_select_postop
                && (cond_dims[head_dim] == 1
                        || cond_strides[head_dim] != conf.ld_scores))
            return false;
        return true;
    };

    // Resolve the BRGEMM row layout once; descriptors and post-ops below
    // are built from it.
    conf.stack_heads = can_stack_heads();
    const int m_axis = conf.stack_heads ? head_dim : row_dim;
    conf.m_block = conf.stack_heads ? conf.query_heads_per_item : q_block;
    conf.mm1_lda = q_strides[m_axis];
    conf.out_ld = out_strides[m_axis];
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
    conf.ldk = need_copy_k ? b_panel_n : k_strides[row_dim];
    conf.ldv = need_copy_v ? b_panel_n : v_strides[row_dim];
    // BRGEMM requires ldb >= hs_v for unpacked V. When seq_kv == 1, V's row
    // stride is unused, so a smaller stride can be raised to hs_v.
    if (!need_copy_v && seq_kv == 1) conf.ldv = nstl::max(conf.ldv, hs_v);
    conf.v_block = need_copy_v ? nstl::min<dim_t>(hs_v, b_panel_n) : hs_v;
    conf.v_tail = hs_v % conf.v_block;

    size_t max_wsp = 0;
    CHECK(build_brgemm_descs(pd, conf, conf.brgemm, max_wsp));
    layout_scratch(desc, mm_dt, max_wsp, conf);

    if (get_verbose(verbose_t::debuginfo) > 0) {
        verbose_printf(verbose_t::debuginfo,
                "sdpa_config: seq_q=%lld seq_kv=%lld hq=%lld hkv=%lld "
                "d_qk=%lld d_v=%lld dt_bytes=%lld q_block=%lld "
                "mm1_block=%lld mm2_block=%lld nthr=%d schedule=%d "
                "mm1_chunks=%lld mm2_chunks=%lld l2_per_core=%u "
                "l3_per_core=%u interleave=%d\n",
                static_cast<long long>(seq_q), static_cast<long long>(seq_kv),
                static_cast<long long>(num_head_q),
                static_cast<long long>(num_head_kv),
                static_cast<long long>(hs_qk), static_cast<long long>(hs_v),
                static_cast<long long>(mm_dt_sz),
                static_cast<long long>(conf.q_block),
                static_cast<long long>(conf.mm1_n_block),
                static_cast<long long>(conf.mm2_k_block), conf.nthr,
                static_cast<int>(conf.schedule),
                static_cast<long long>(conf.split_kv.mm1_chunks),
                static_cast<long long>(conf.split_kv.mm2_chunks),
                platform::get_per_core_cache_size(2),
                platform::get_per_core_cache_size(3),
                static_cast<int>(conf.interleave_items));
    }

    return status::success;
}

status_t create_kernels(cpu_isa_t isa_for_ir, const sdpa_pd_t &pd,
        const sdpa_full_softmax_conf_t &conf, engine_t *engine,
        sdpa_full_softmax_kernels_t &kernels) {
    constexpr int ndims = sdpa_pd_t::ndims;
    constexpr int row_dim = ndims - 2;

    const sdpa_desc_t &desc = *pd.desc();
    const dim_t seq_kv = desc.keys();
    const dim_t hs_qk = desc.head_size();
    const dim_t hs_v = desc.values();
    const data_type_t mm_dt = mm_data_type(pd);
    const dims_t &k_strides = memory_desc_wrapper(desc.key_md()).strides();
    const dims_t &v_strides = memory_desc_wrapper(desc.val_md()).strides();

    // Packers for B = [rows x cols] with the given source strides. Packing ahead
    // packs the whole tensor; packing per tile packs one full block of the
    // tiled axis ([0]) or its tail ([1]).
    auto init_packers
            = [&](std::unique_ptr<sdpa_brgemm_copy::b_packer_t>(&packers)[2],
                      const dim_t(&rows)[2], const dim_t(&cols)[2],
                      dim_t row_stride, dim_t col_stride) -> status_t {
        for (int i = 0; i < 2; ++i) {
            if (!rows[i] || !cols[i]) continue;
            packers[i].reset(new sdpa_brgemm_copy::b_packer_t());
            CHECK(packers[i]->init(
                    rows[i], cols[i], row_stride, col_stride, mm_dt));
        }
        return status::success;
    };
    const bool per_tile
            = conf.kv_packing == sdpa_full_softmax_kv_packing_t::per_tile;
    if (conf.pack_k) {
        const dim_t rows[2] = {hs_qk, hs_qk};
        const dim_t cols[2] = {per_tile ? conf.mm1_n_block : seq_kv,
                per_tile ? conf.mm1_n_tail : 0};
        CHECK(init_packers(kernels.k_packers, rows, cols, k_strides[row_dim],
                k_strides[ndims - 1]));
    }
    if (conf.pack_v) {
        const dim_t rows[2] = {per_tile ? conf.mm2_k_block : seq_kv,
                per_tile ? conf.mm2_k_tail : 0};
        const dim_t cols[2] = {hs_v, hs_v};
        CHECK(init_packers(kernels.v_packers, rows, cols, v_strides[row_dim],
                v_strides[ndims - 1]));
    }
    if (conf.schedule == sdpa_full_softmax_schedule_t::split_kv
            && conf.split_kv.mm2_chunks > 1) {
        kernels.split_kv_acc.reset(new cpu_accumulator_1d_t<data_type::f32>());
        CHECK(kernels.split_kv_acc->create_kernel());
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
        dims_t sm_dims = {conf.q_block, seq_kv};
        CHECK(memory_desc_init_by_tag(
                sm_src_md, 2, sm_dims, data_type::f32, format_tag::ab));
        CHECK(memory_desc_init_by_tag(
                sm_dst_md, 2, sm_dims, mm_dt, format_tag::ab));
        softmax_desc_t sd {};
        sd.primitive_kind = primitive_kind::softmax;
        sd.prop_kind = prop_kind::forward_inference;
        // inf_as_zero: a fully-masked row (all -inf) must produce an
        // all-zero row instead of NaN. The jit softmax kernel implements
        // this natively for the softmax_accurate_inf_as_zero alg.
        sd.alg_kind = desc.softmax_alg == alg_kind::softmax_accurate_inf_as_zero
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
                kernels.softmax_pd = *it;
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
        kernels.softmax_kernel.reset(k);
    }

    // Build the standalone select-mask pre-pass kernel when a select mask is
    // present and is not already folded into mm1. Polarity, the seq_kv width,
    // and the score/condition row strides are baked in; the row count is a
    // runtime argument, so one instance serves both the full and query-tail
    // tiles. The condition is dense along seq_kv (cond_col == 1) because the
    // pd requires the key axis to be full and plain.
    if (pd.with_select_mask() && !conf.mm1_select_postop) {
        const memory_desc_wrapper cond_mdw(desc.attn_mask_md());
        const dim_t cond_row = cond_mdw.dims()[sdpa_pd_t::mask_q_index] == 1
                ? 0
                : cond_mdw.strides()[sdpa_pd_t::mask_q_index];
        std::shared_ptr<jit_generator_t> k(
                sdpa_full_softmax_select_ir::create_select_ir_kernel(isa_for_ir,
                        /*wseq_kv=*/seq_kv,
                        /*invert_select=*/desc.invert_select,
                        /*scores_row_stride=*/conf.ld_scores,
                        /*cond_row_stride=*/cond_row));
        CHECK(k->create_kernel());
        kernels.select_kernel = std::move(k);
    }

    return status::success;
}

namespace {

// Inputs of one execute() call, shared by all of its tasks.
struct exec_state_t {
    exec_state_t(const sdpa_pd_t &pd, const sdpa_full_softmax_conf_t &conf,
            const sdpa_full_softmax_kernels_t &kernels,
            const sdpa_full_softmax_run_args_t &args, int nthr)
        : pd(pd)
        , desc(*pd.desc())
        , conf(conf)
        , kernels(kernels)
        , args(args)
        , out_d(pd.dst_md())
        , mm_dt(mm_data_type(pd))
        , mm_dt_sz(types::data_type_size(mm_dt))
        , scratch(static_cast<char *>(args.scratch))
        , shared_area_offset(conf.scratch_per_thread * nthr)
        , k_td(args.k, desc.key_md(), conf.k_copy_desc, scratch,
                  shared_area_offset)
        , v_td(args.v, desc.val_md(), conf.v_copy_desc, scratch,
                  shared_area_offset) {}

    char *thread_scratch(int tid) const {
        return scratch + tid * conf.scratch_per_thread;
    }
    char *shared_area() const { return scratch + shared_area_offset; }

    // GQA group size: query heads per KV head (>= 1).
    dim_t group_head() const {
        return desc.num_q_heads() / desc.num_kv_heads();
    }

    // Query rows of a query head from query q0.
    const char *q(dim_t batch_idx, dim_t head, dim_t q0) const {
        return get_arg_ptr(args.q, desc.qry_md(), batch_idx, head, q0);
    }
    // Output rows of a query head from query q0.
    char *out(dim_t batch_idx, dim_t head, dim_t q0) const {
        return get_arg_ptr(args.out, out_d, batch_idx, head, q0);
    }
    // Select condition rows of a query head from query q0, or null without a
    // select.
    const uint8_t *cond(dim_t batch_idx, dim_t head, dim_t q0) const {
        if (!pd.with_select_mask()) return nullptr;
        return reinterpret_cast<const uint8_t *>(get_arg_ptr(
                args.cond, desc.attn_mask_md(), batch_idx, head, q0));
    }

    const sdpa_pd_t &pd;
    const sdpa_desc_t &desc;
    const sdpa_full_softmax_conf_t &conf;
    const sdpa_full_softmax_kernels_t &kernels;
    const sdpa_full_softmax_run_args_t &args;
    const memory_desc_wrapper out_d;
    const data_type_t mm_dt;
    const size_t mm_dt_sz;
    char *const scratch;
    const size_t shared_area_offset;
    const b_tensor_desc_t k_td, v_td;
};

// When packing ahead, materialise mm1's B (K as a dense [hs_qk, seq_kv]
// matrix; the brgemm ukernel has no transposed-B mode) and mm2's B (V) once
// per (batch, kv_head) into the shared area.
void pack_kv_ahead(const exec_state_t &st, int nthr) {
    const dim_t num_head_kv = st.desc.num_kv_heads();
    const bool pack_k = st.k_td.copy_desc;
    const bool pack_v = st.v_td.copy_desc;
    if (!pack_k && !pack_v) return;

    auto chunk_rows_of = [&](const sdpa_brgemm_copy::b_packer_t &p) {
        // Rows per fine-grained task: about 128 (an untuned target), rounded up to
        // the packer's K step.
        constexpr dim_t target_chunk_rows = 128;
        return utils::rnd_up(target_chunk_rows, p.k_step());
    };
    const dim_t k_chunk_rows
            = pack_k ? chunk_rows_of(*st.kernels.k_packers[0]) : 0;
    const dim_t v_chunk_rows
            = pack_v ? chunk_rows_of(*st.kernels.v_packers[0]) : 0;

    const dim_t batch_head_count = st.desc.batch() * num_head_kv;
    const dim_t coarse_work = batch_head_count * (pack_k + pack_v);
    const bool fine_grained = coarse_work < nthr;

    constexpr dim_t packer_panel_n = sdpa_brgemm_copy::b_packer_t::panel_n;
    const dim_t k_n_panels
            = fine_grained ? utils::div_up(st.desc.keys(), packer_panel_n) : 1;
    const dim_t k_k_chunks = fine_grained && pack_k
            ? utils::div_up(st.desc.head_size(), k_chunk_rows)
            : 1;
    const dim_t v_n_panels = fine_grained
            ? utils::div_up(st.desc.values(), packer_panel_n)
            : 1;
    const dim_t v_k_chunks = fine_grained && pack_v
            ? utils::div_up(st.desc.keys(), v_chunk_rows)
            : 1;
    const dim_t k_work
            = pack_k ? batch_head_count * k_n_panels * k_k_chunks : 0;
    const dim_t v_work
            = pack_v ? batch_head_count * v_n_panels * v_k_chunks : 0;
    const dim_t work = k_work + v_work;

    // Keep whole-head tasks unless they cannot fill the team.
    parallel(adjust_num_threads(nthr, work), [&](int ithr, int nthr) {
        dim_t start = 0, end = 0;
        balance211(work, nthr, ithr, start, end);
        for (dim_t task = start; task < end; ++task) {
            const bool is_k = task < k_work;
            const dim_t operand_task = is_k ? task : task - k_work;
            const dim_t n_panels = is_k ? k_n_panels : v_n_panels;
            const dim_t k_chunks = is_k ? k_k_chunks : v_k_chunks;
            const dim_t head_idx = operand_task / (k_chunks * n_panels);

            const auto &td = is_k ? st.k_td : st.v_td;
            const auto &packer = is_k ? *st.kernels.k_packers[0]
                                      : *st.kernels.v_packers[0];

            const dim_t batch_idx = head_idx / num_head_kv;
            const dim_t kv_head_idx = head_idx % num_head_kv;
            const char *src
                    = get_arg_ptr(td.buffer, td.md, batch_idx, kv_head_idx);
            char *dst = get_copy_ptr(td, batch_idx, kv_head_idx);

            if (!fine_grained) {
                packer.execute(src, dst);
                continue;
            }

            const dim_t n_panel_idx = operand_task / k_chunks % n_panels;
            const dim_t k_chunk_idx = operand_task % k_chunks;

            const dim_t n0 = n_panel_idx * packer_panel_n;
            const dim_t chunk_rows = is_k ? k_chunk_rows : v_chunk_rows;
            const dim_t k0 = k_chunk_idx * chunk_rows;

            const dim_t k_dim_size
                    = is_k ? st.desc.head_size() : st.desc.keys();
            const dim_t k1 = nstl::min(k0 + chunk_rows, k_dim_size);
            packer.execute_panel_range(src, dst, n0, k0, k1);
        }
    });
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
    const bool is_f32 = st.mm_dt == data_type::f32;
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
        sp.process_n_elems = static_cast<size_t>(st.desc.keys());
        sp.dst_orig = prob_row;
        sp.post_ops_binary_rhs_arg_vec = nullptr;
        (*st.kernels.softmax_kernel)(&sp);
    }
}

// State of one thread of a parallel region, shared by all of the thread's
// work items: its scratch, the pointer vectors (allocated once), and the AMX
// palette currently loaded. Configured AMX tiles are released on destruction,
// i.e. once per thread at the end of the region.
class thread_state_t {
public:
    thread_state_t(const exec_state_t &st, int tid)
        : scratch(st.thread_scratch(tid))
        , amx_wsp(st.conf.amx_wsp_bytes > 0
                          ? scratch + st.conf.thread_scratch.amx_wsp
                          : nullptr) {
        heads.reserve(st.conf.query_heads_per_item);
        // Upper bound: one rhs per mm1 post-op, plus fill and condition.
        rhs.reserve(st.conf.query_heads_per_item
                * (st.conf.mm1_post_ops.size() + 2));
    }

    ~thread_state_t() {
        if (used_amx) amx_tile_release();
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(thread_state_t);

    struct head_ptrs_t {
        const char *q;
        char *out;
        const uint8_t *cond;
    };

    char *const scratch;
    void *const amx_wsp;
    std::vector<head_ptrs_t> heads;
    std::vector<const void *> rhs;
    int prev_brg_idx = -1;
    bool used_amx = false;
};

// One work item: query block q_block_idx of the query_heads_per_item
// consecutive query heads starting at item_head_idx * query_heads_per_item,
// which share one KV head. Its operations take a key range and the tiles to
// use, so a work item serves both a whole fused computation in per-thread
// tiles and one split_kv chunk in shared tiles. It uses the thread's
// thread_state_t for its pointers and the AMX tile configuration.
class work_item_t {
public:
    work_item_t(const exec_state_t &st, thread_state_t &thread_state,
            dim_t batch_idx, dim_t item_head_idx, dim_t q_block_idx)
        : st_(st)
        , conf_(st.conf)
        , thread_state_(thread_state)
        , batch_idx_(batch_idx)
        , kv_head_idx_(
                  item_head_idx * conf_.query_heads_per_item / st.group_head())
        , m_(nstl::min(conf_.q_block,
                  st.desc.queries() - q_block_idx * conf_.q_block))
        , qi_(m_ != conf_.q_block)
        , call_m_(qi_ ? conf_.q_tail : conf_.m_block)
        // Stacked heads: one BRGEMM call covers all heads of the work item,
        // with head 0's pointers and the head stride as the row stride.
        , brgemm_calls_(conf_.stack_heads ? 1 : conf_.query_heads_per_item)
        , has_mm1_postops_(
                  !st.conf.mm1_post_ops.empty() || conf_.mm1_select_postop) {
        const dim_t head0 = item_head_idx * conf_.query_heads_per_item;
        const dim_t q0 = q_block_idx * conf_.q_block;
        auto &heads = thread_state_.heads;
        auto &rhs = thread_state_.rhs;
        heads.resize(conf_.query_heads_per_item);
        rhs.clear();
        for (dim_t h = 0; h < conf_.query_heads_per_item; ++h) {
            const dim_t head = head0 + h;
            auto &hp = heads[h];
            hp.q = st.q(batch_idx, head, q0);
            hp.out = st.out(batch_idx, head, q0);
            hp.cond = st.cond(batch_idx, head, q0);
            if (has_mm1_postops_)
                append_mm1_rhs(rhs, st.conf.mm1_post_ops, st.args, batch_idx,
                        head, q0, conf_.mm1_select_postop, hp.cond);
        }
        rhs_per_head_ = rhs.size() / conf_.query_heads_per_item;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(work_item_t);

    // The user output of the item's query heads, for mm2 writing to it
    // directly.
    item_tiles_t output_tiles() const {
        const size_t head_stride
                = st_.out_d.strides()[1] * st_.out_d.data_type_size();
        return {thread_state_.heads[0].out, head_stride};
    }

    // mm1: scores[m, kv_begin:kv_end) = Q[m, hs_qk] * K[hs_qk, kv_begin:kv_end)
    // in mm1_n_block-wide column blocks (+ tail). Each call writes its columns
    // of the full score tile (LDC = ld_scores). The post-op injector addresses
    // the mask or select condition at the global column: data_C_ptr_ is the
    // full-tile base, while c_blk is offset by the column.
    void mm1(dim_t kv_begin, dim_t kv_end, const item_tiles_t &scores) {
        const auto &brg = conf_.brgemm;
        b_operand_t k_op(st_.k_td, batch_idx_, kv_head_idx_,
                st_.kernels.k_packers, conf_.k_tile_desc, thread_state_.scratch,
                st_.mm_dt_sz);
        for_each_block(kv_begin, kv_end, conf_.mm1_n_block,
                [&](dim_t kv0, bool n_tail) {
            k_op.load_tile(0, kv0, n_tail);
            const brgemm_kernel_t *kernel = this->kernel(brg.mm1[qi_][n_tail]);
            for (dim_t h = 0; h < brgemm_calls_; ++h) {
                brgemm_batch_element_t be;
                be.ptr.A = thread_state_.heads[h].q;
                be.ptr.B = k_op.at(0, kv0);
                float *c_blk = scores.f32(h) + kv0;
                if (has_mm1_postops_) {
                    brgemm_post_ops_data_t pod(/*bias=*/nullptr,
                            /*binary_post_ops_rhs=*/thread_state_.rhs.data()
                                    + h * rhs_per_head_,
                            /*oc_logical_off=*/0,
                            /*dst_row_logical_off=*/0,
                            /*data_C_ptr_=*/scores.of(h),
                            /*first_mb_matrix_addr_off=*/0);
                    brgemm_kernel_execute_postops(kernel, 1, &be, c_blk, c_blk,
                            pod, thread_state_.amx_wsp);
                } else {
                    brgemm_kernel_execute(
                            kernel, 1, &be, c_blk, thread_state_.amx_wsp);
                }
            }
        });
    }

    // Softmax of the item's score rows; see softmax_rows().
    void softmax(const item_tiles_t &scores, const item_tiles_t &prob) const {
        for (dim_t h = 0; h < conf_.query_heads_per_item; ++h)
            softmax_rows(st_, scores.f32(h), prob.of(h),
                    thread_state_.heads[h].cond, m_);
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
                st_.kernels.v_packers, conf_.v_tile_desc, thread_state_.scratch,
                st_.mm_dt_sz);
        for_each_block(kv_begin, kv_end, conf_.mm2_k_block,
                [&](dim_t kv0, bool k_tail) {
            v_op.load_tile(kv0, 0, k_tail);
            const bool use_beta0_kernel
                    = conf_.mm2_beta != 0.0f && kv0 == kv_begin;
            const bool last_k_block = kv0 + conf_.mm2_k_block >= kv_end;
            const bool apply_postops = conf_.mm2_postops && last_k_block;
            assert(!(use_beta0_kernel && k_tail));
            for_each_block(0, st_.desc.values(), conf_.v_block,
                    [&](dim_t v0, bool v_tail) {
                const brgemm_kernel_t *kernel = this->kernel(use_beta0_kernel
                                ? brg.mm2_beta0[qi_][v_tail]
                                : brg.mm2[qi_][k_tail][v_tail]);
                for (dim_t h = 0; h < brgemm_calls_; ++h) {
                    brgemm_batch_element_t be;
                    be.ptr.A = prob.of(h) + kv0 * st_.mm_dt_sz;
                    be.ptr.B = v_op.at(kv0, v0);
                    float *c_blk = c.f32(h) + v0;
                    if (apply_postops) {
                        char *dst = thread_state_.heads[h].out
                                + v0 * st_.out_d.data_type_size();
                        const brgemm_post_ops_data_t pod;
                        brgemm_kernel_execute_postops(kernel, 1, &be, c_blk,
                                dst, pod, thread_state_.amx_wsp);
                    } else {
                        brgemm_kernel_execute(
                                kernel, 1, &be, c_blk, thread_state_.amx_wsp);
                    }
                }
            });
        });
    }

    // Writes the item's f32 mm2 result tiles to the user output.
    void store(const item_tiles_t &pv) const {
        const sdpa_desc_t &desc = st_.desc;
        const dim_t out_col_stride = st_.out_d.strides()[sdpa_pd_t::ndims - 1];
        for (dim_t h = 0; h < brgemm_calls_; ++h)
            store_pv_tile(pv.f32(h), conf_.ld_pv, thread_state_.heads[h].out,
                    conf_.out_ld, out_col_stride, st_.out_d.data_type(),
                    call_m_, desc.values());
    }

private:
    // Returns the kernel of descriptor brg_idx, configuring AMX tiles if it
    // needs a different palette than the previous call.
    const brgemm_kernel_t *kernel(int brg_idx) {
        assert(brg_idx >= 0);
        const bool is_amx = (*conf_.brgemm.descs)[brg_idx]->is_tmm;
        st_.kernels.brgemm_palettes.maybe_tile_configure(
                is_amx, thread_state_.prev_brg_idx, brg_idx);
        thread_state_.used_amx = thread_state_.used_amx || is_amx;
        const brgemm_kernel_t *kernel = st_.kernels.brgemm_kernels[brg_idx];
        assert(kernel);
        return kernel;
    }

    const exec_state_t &st_;
    const sdpa_full_softmax_conf_t &conf_;
    thread_state_t &thread_state_;
    const dim_t batch_idx_, kv_head_idx_;
    // Query rows of each head, and whether they form the query tail block.
    const dim_t m_;
    const int qi_;
    // Rows of each BRGEMM call, and calls per mm1/mm2 block.
    const dim_t call_m_, brgemm_calls_;
    const bool has_mm1_postops_;
    size_t rhs_per_head_ = 0;
};

// per_head and kv_head_group: each work item computes all keys of its query
// block in the thread's score, P and pv tiles.
void execute_fused(const exec_state_t &st, int nthr) {
    const auto &conf = st.conf;
    const sdpa_desc_t &desc = st.desc;
    const auto &scratch_layout = conf.thread_scratch;
    const size_t q_block = static_cast<size_t>(conf.q_block);
    const dim_t n_qblk = utils::div_up(desc.queries(), conf.q_block);
    const dim_t n_items = desc.num_q_heads() / conf.query_heads_per_item;
    auto body = [&](thread_state_t &thread_state, dim_t batch_idx,
                        dim_t item_head_idx, dim_t q_block_idx) {
        const item_tiles_t scores = {
                thread_state.scratch, q_block * conf.ld_scores * sizeof(float)};
        const item_tiles_t prob = scratch_layout.has_prob
                ? item_tiles_t {thread_state.scratch + scratch_layout.prob,
                          q_block * conf.ld_prob * st.mm_dt_sz}
                : scores;

        work_item_t item(
                st, thread_state, batch_idx, item_head_idx, q_block_idx);
        item.mm1(0, desc.keys(), scores);
        item.softmax(scores, prob);
        if (conf.mm2_direct) {
            item.mm2(0, desc.keys(), prob, item.output_tiles());
        } else {
            const item_tiles_t pv = {thread_state.scratch + scratch_layout.pv,
                    q_block * conf.ld_pv * sizeof(float)};
            item.mm2(0, desc.keys(), prob, pv);
            if (!conf.mm2_postops) item.store(pv);
        }
    };
    const dim_t work = desc.batch() * n_items * n_qblk;
    parallel(adjust_num_threads(nthr, work), [&](int ithr, int nthr) {
        thread_state_t thread_state(st, ithr);
        if (conf.interleave_items) {
            for (dim_t i = ithr; i < work; i += nthr)
                body(thread_state, i / (n_items * n_qblk),
                        (i / n_qblk) % n_items, i % n_qblk);
            return;
        }
        for_nd(ithr, nthr, desc.batch(), n_items, n_qblk,
                [&](dim_t batch_idx, dim_t item_head_idx, dim_t q_block_idx) {
            body(thread_state, batch_idx, item_head_idx, q_block_idx);
        });
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
    const sdpa_desc_t &desc = st.desc;
    const auto &sk = conf.split_kv;
    const dim_t n_items = desc.num_q_heads() / conf.query_heads_per_item;
    const dim_t n_qblk = utils::div_up(desc.queries(), conf.q_block);
    const dim_t item_work = desc.batch() * n_items * n_qblk;

    const size_t score_row_bytes = conf.ld_scores * sizeof(float);
    const size_t prob_row_bytes = conf.thread_scratch.has_prob
            ? conf.ld_prob * st.mm_dt_sz
            : score_row_bytes;
    const size_t pv_row_bytes = conf.ld_pv * sizeof(float);
    const size_t rows = static_cast<size_t>(desc.batch()) * desc.num_q_heads()
            * desc.queries();
    const size_t partial_bytes = rows * pv_row_bytes;

    auto row_index = [&](dim_t batch_idx, dim_t head, dim_t q0) {
        return static_cast<size_t>(
                (batch_idx * desc.num_q_heads() + head) * desc.queries() + q0);
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
                desc.queries() * row_bytes};
    };
    const dim_t mm1_work = item_work * sk.mm1_chunks;
    parallel(adjust_num_threads(nthr, mm1_work), [&](int ithr, int nthr) {
        thread_state_t thread_state(st, ithr);
        for_nd(ithr, nthr, desc.batch(), n_items, n_qblk, sk.mm1_chunks,
                [&](dim_t batch_idx, dim_t item_head_idx, dim_t q_block_idx,
                        dim_t c) {
            work_item_t item(
                    st, thread_state, batch_idx, item_head_idx, q_block_idx);
            const auto kv = split_kv_chunk_keys(
                    c, sk.mm1_chunks, conf.mm1_n_block, desc.keys());
            item.mm1(kv.first, kv.second,
                    item_tiles(sk.scores, score_row_bytes, batch_idx,
                            item_head_idx, q_block_idx));
        });
    });

    parallel_nd_ext(nthr, desc.batch(), desc.num_q_heads(), n_qblk,
            [&](int, int, dim_t batch_idx, dim_t head, dim_t q_block_idx) {
        const dim_t q0 = q_block_idx * conf.q_block;
        softmax_rows(st,
                reinterpret_cast<float *>(shared_rows(
                        sk.scores, score_row_bytes, batch_idx, head, q0)),
                shared_rows(sk.prob, prob_row_bytes, batch_idx, head, q0),
                st.cond(batch_idx, head, q0),
                nstl::min(conf.q_block, desc.queries() - q0));
    });

    const dim_t mm2_work = item_work * sk.mm2_chunks;
    parallel(adjust_num_threads(nthr, mm2_work), [&](int ithr, int nthr) {
        thread_state_t thread_state(st, ithr);
        for_nd(ithr, nthr, desc.batch(), n_items, n_qblk, sk.mm2_chunks,
                [&](dim_t batch_idx, dim_t item_head_idx, dim_t q_block_idx,
                        dim_t c) {
            work_item_t item(
                    st, thread_state, batch_idx, item_head_idx, q_block_idx);
            const auto kv = split_kv_chunk_keys(
                    c, sk.mm2_chunks, conf.mm2_k_block, desc.keys());
            const auto c_tile = conf.mm2_direct
                    ? item.output_tiles()
                    : item_tiles(sk.partials + c * partial_bytes, pv_row_bytes,
                              batch_idx, item_head_idx, q_block_idx);
            item.mm2(kv.first, kv.second,
                    item_tiles(sk.prob, prob_row_bytes, batch_idx,
                            item_head_idx, q_block_idx),
                    c_tile);
            if (sk.mm2_chunks == 1 && !conf.mm2_direct && !conf.mm2_postops)
                item.store(c_tile);
        });
    });
    if (sk.mm2_chunks == 1) return;

    const dim_t out_row_stride = st.out_d.strides()[sdpa_pd_t::ndims - 2];
    const dim_t out_col_stride = st.out_d.strides()[sdpa_pd_t::ndims - 1];
    const size_t partial_floats = partial_bytes / sizeof(float);
    const bool fuse_add_and_convert_to_xf16 = out_col_stride == 1
            && utils::one_of(
                    st.out_d.data_type(), data_type::bf16, data_type::f16);
    auto &acc_ker = *st.kernels.split_kv_acc;
    parallel_nd_ext(nthr, desc.batch(), desc.num_q_heads(), n_qblk,
            [&](int, int, dim_t batch_idx, dim_t head, dim_t q_block_idx) {
        const dim_t q0 = q_block_idx * conf.q_block;
        const dim_t m = nstl::min(conf.q_block, desc.queries() - q0);
        float *acc = reinterpret_cast<float *>(
                shared_rows(sk.partials, pv_row_bytes, batch_idx, head, q0));

        const dim_t last_chunk = sk.mm2_chunks - 1;
        for (dim_t c = 1; c < last_chunk; ++c) {
            const float *src = acc + c * partial_floats;
            add_f32_tile(acc_ker, acc, src, m, desc.values(), conf.ld_pv);
        }

        char *out = st.out(batch_idx, head, q0);
        if (fuse_add_and_convert_to_xf16) {
            add_tile_and_cvt_to_xf16(acc, acc + last_chunk * partial_floats,
                    conf.ld_pv, out, out_row_stride, st.out_d.data_type(), m,
                    desc.values());
        } else {
            const float *src = acc + last_chunk * partial_floats;
            add_f32_tile(acc_ker, acc, src, m, desc.values(), conf.ld_pv);

            store_pv_tile(acc, conf.ld_pv, out, out_row_stride, out_col_stride,
                    st.out_d.data_type(), m, desc.values());
        }
    });
}

} // namespace

status_t execute(const sdpa_pd_t &pd, const sdpa_full_softmax_conf_t &conf,
        const sdpa_full_softmax_kernels_t &kernels,
        const sdpa_full_softmax_run_args_t &args, int nthr) {
    const exec_state_t st(pd, conf, kernels, args, nthr);
    if (conf.kv_packing == sdpa_full_softmax_kv_packing_t::ahead)
        pack_kv_ahead(st, nthr);
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
