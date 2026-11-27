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
#include <cstdint>
#include <limits>

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/sdpa/sdpa_online_softmax.hpp"
#include "cpu/x64/sdpa/sdpa_online_softmax_ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
inline size_t align64(size_t n) {
    return utils::rnd_up(n, size_t(64));
}
} // namespace

sdpa_online_softmax_kernels_t::sdpa_online_softmax_kernels_t() = default;

sdpa_online_softmax_kernels_t::~sdpa_online_softmax_kernels_t() {
    for (int qi = 0; qi < 2; ++qi)
        for (int ki = 0; ki < 2; ++ki) {
            if (mm1_kernel[qi][ki]) brgemm_kernel_destroy(mm1_kernel[qi][ki]);
            if (mm2_kernel[qi][ki]) brgemm_kernel_destroy(mm2_kernel[qi][ki]);
        }
}

namespace sdpa_online_softmax {

status_t configure(const sdpa_pd_t &pd, sdpa_online_softmax_conf_t &conf) {
    conf = sdpa_online_softmax_conf_t();

    const sdpa_desc_t &desc = *pd.desc();
    const dim_t seq_q = desc.queries();
    const dim_t seq_kv = desc.keys();
    const dim_t hs_qk = desc.head_size();
    const dim_t hs_v = desc.values();

    const dim_t l2_budget_bytes
            = static_cast<dim_t>(3 * platform::get_per_core_cache_size(2) / 4);

    // KV tile width. Each KV tile runs mm1 -> softmax -> mm2 over a kv_blk-wide
    // column slice. kv_blk is a single shared value, not one per matmul: it is
    // the contraction link between the two brgemms -- mm1 emits scores[m, kv_blk]
    // (kv_blk is mm1's N), softmax rewrites that same tile in place, and mm2
    // contracts it as P[m, kv_blk] * V (kv_blk is mm2's K). So mm1's N *is*
    // mm2's K by construction and the two cannot use different widths.
    // The two matmul B operands -- mm1's K panel [hs_qk x kv_blk]
    // and mm2's V panel [kv_blk x hs_v] -- are each re-read once per q_blk
    // M-block of their brgemm, so they must stay L2-resident across those
    // re-reads or the inner loop goes memory-bound. Their footprint is
    // (hs_qk|hs_v) * kv_blk, i.e. it grows with the head size; to hold the
    // footprint (and thus the L2 pressure) constant across head sizes, kv_blk
    // must scale as 1/head_size. Pin the wider panel to L2/8 (~192 KiB here) so
    // the two panels together take ~L2/4 and leave ~3/4 L2 for the q_blk-scaled
    // state sized below. This 1/head-size law is empirically the per-head
    // optimum (hs 512 -> kv 64, hs 128 -> kv 384, hs 64 -> kv 768); at the small
    // end the panel also drops into L1. Round DOWN to a column granule that
    // keeps clean AVX512 f32 vectorization (a multiple of the 16-lane vector;
    // this path is always f32, never AMX): prefer a 64-wide block (4 zmm, best
    // mm1 N-register use) when the budget allows it, else degrade to 32, else a
    // 16-column floor -- so a large head or a small L2 shrinks the block rather
    // than collapsing it to the whole (untiled) axis. Clamp to seq_kv (short
    // context runs untiled).
    const dim_t b_panel_budget_bytes = l2_budget_bytes / 8;
    const dim_t b_panel_rows = nstl::max(hs_qk, hs_v);
    dim_t kv_block_width = b_panel_rows > 0
            ? static_cast<dim_t>(b_panel_budget_bytes
                      / (b_panel_rows * static_cast<dim_t>(sizeof(float))))
            : seq_kv;
    const dim_t granule
            = kv_block_width >= 64 ? 64 : (kv_block_width >= 32 ? 32 : 16);
    kv_block_width = nstl::max<dim_t>(
            utils::rnd_dn(kv_block_width, granule), (dim_t)16);
    conf.kv_blk = nstl::min<dim_t>(seq_kv, kv_block_width);

    // Query block. Per KV tile mm1 re-reads the Q slice [q_blk x hs_qk] and the
    // acc-renorm re-reads+writes the running acc [q_blk x hs_v]; these carry
    // across the whole KV sweep, so holding them cache-resident is what turns
    // mm1 from memory-bound (re-reading Q every tile) into compute-bound, and a
    // larger q_blk also amortizes the once-per-block K/V stream. Size q_blk from
    // the per-row state -- Q [hs_qk], acc + pv [hs_v] each, one live scores row
    // [kv_blk] -- against the full 3/4-L2 budget. The B panels (another ~L2/4,
    // reserved above via kv_blk) are deliberately NOT subtracted here: acc/pv/Q
    // are swept sequentially and tolerate spilling to L3, whereas the panels are
    // re-read and must stay put -- so the panels get first claim on L2 and q_blk
    // is allowed to overcommit the remainder. Clamp to [1, seq_q].
    const dim_t per_row_bytes = static_cast<dim_t>(
            (hs_qk + 2 * hs_v + conf.kv_blk) * sizeof(float));
    conf.q_blk = utils::saturate((dim_t)1, seq_q,
            static_cast<dim_t>(l2_budget_bytes / per_row_bytes));
    conf.q_tail = seq_q % conf.q_blk;

    // Per-thread scratch: scores, acc, pv, row_max, row_denom, old_coef, each
    // 64-byte aligned so no buffer straddles a cacheline shared with the next.
    // Sized for one query block (q_blk rows), not the full seq_q. Pure
    // arithmetic (no kernel dependency), so a primitive_desc can size its
    // scratchpad from configure() alone.
    const size_t fsz = sizeof(float);
    const size_t scores_bytes
            = align64(static_cast<size_t>(conf.q_blk) * conf.kv_blk * fsz);
    const size_t acc_bytes
            = align64(static_cast<size_t>(conf.q_blk) * hs_v * fsz);
    const size_t pv_bytes
            = align64(static_cast<size_t>(conf.q_blk) * hs_v * fsz);
    const size_t row_bytes = align64(static_cast<size_t>(conf.q_blk) * fsz);
    conf.off_scores = 0;
    conf.off_acc = conf.off_scores + scores_bytes;
    conf.off_pv = conf.off_acc + acc_bytes;
    conf.off_row_max = conf.off_pv + pv_bytes;
    conf.off_row_denom = conf.off_row_max + row_bytes;
    conf.off_old_coef = conf.off_row_denom + row_bytes;
    conf.scratch_per_thread = conf.off_old_coef + row_bytes;

    conf.nthr = dnnl_get_max_threads();

    return status::success;
}

status_t create_kernels(cpu_isa_t isa_for_ir, const sdpa_pd_t &pd,
        const sdpa_online_softmax_conf_t &conf, engine_t *engine,
        sdpa_online_softmax_kernels_t &kernels) {
    UNUSED(engine);
    constexpr int ndims = sdpa_pd_t::ndims;
    constexpr int row_dim = ndims - 2;

    const sdpa_desc_t &desc = *pd.desc();
    const dim_t seq_kv = desc.keys();
    const dim_t hs_qk = desc.head_size();
    const dim_t hs_v = desc.values();
    const dims_t &q_strides = memory_desc_wrapper(desc.qry_md()).strides();
    const dims_t &k_strides = memory_desc_wrapper(desc.key_md()).strides();
    const dims_t &v_strides = memory_desc_wrapper(desc.val_md()).strides();
    const bool has_select = pd.with_select_mask();

    const dim_t kv_tail = seq_kv % conf.kv_blk;
    // Query-block row counts indexed by the q-tail flag: [0] full block, [1]
    // ragged last block (only built when q_tail != 0).
    const dim_t q_rows[2] = {conf.q_blk, conf.q_tail};
    // KV tile widths indexed by the kv-tail flag.
    const dim_t kv_w[2] = {conf.kv_blk, kv_tail};

    // Create the BRGEMM kernels. Shapes differ only in the M (query-block rows)
    // and N/K (KV tile width) dims, so one kernel per (q full/tail, kv
    // full/tail) combination suffices.
    //   mm1 (beta=0): scores_tile[m, w] = Q[m, hs_qk] * K[hs_qk, w]
    //   mm2 (beta=0): pv_tile[m, hs_v]  = P_tile[m, w] * V[w, hs_v]
    // where m is the query-block rows (q_blk or the seq_q % q_blk remainder)
    // and w is the KV tile width (kv_blk or the seq_kv % kv_blk remainder).
    auto create_brgemm
            = [&](brgemm_kernel_t **out, float beta, dim_t M, dim_t N, dim_t K,
                      dim_t lda, dim_t ldb, dim_t ldc) -> status_t {
        brgemm_desc_t brg;
        CHECK(brgemm_desc_init(&brg, isa_undef, brgemm_addr,
                dnnl::impl::data_type::f32, dnnl::impl::data_type::f32,
                /*transA=*/false, /*transB=*/false, brgemm_row_major,
                /*alpha=*/1.0f, beta, lda, ldb, ldc, M, N, K,
                /*strides=*/nullptr));
        CHECK(brgemm_desc_finalize(&brg));
        brgemm_kernel_t *k = nullptr;
        CHECK(brgemm_kernel_create(&k, brg));
        *out = k;
        return status::success;
    };

    // mm1 writes a dense [m, w] tile (ldc = w); mm2 multiplies that dense tile
    // by V into a dense [m, hs_v] per-tile buffer (beta=0). The running
    // normalized output is combined in the epilogue, so magnitudes stay O(|V|)
    // (matches the full-softmax path's normalize-before accuracy).
    auto create_tile_kernels = [&](brgemm_kernel_t **mm1, brgemm_kernel_t **mm2,
                                       dim_t m, dim_t w) {
        CHECK(create_brgemm(mm1, /*beta=*/0.0f, m, w, hs_qk,
                /*lda=*/q_strides[row_dim],
                /*ldb=*/k_strides[row_dim],
                /*ldc=*/w));
        CHECK(create_brgemm(mm2, /*beta=*/0.0f, m, hs_v, w,
                /*lda=*/w, /*ldb=*/v_strides[row_dim], /*ldc=*/hs_v));
        return status::success;
    };

    const int n_qi = conf.q_tail != 0 ? 2 : 1;
    const int n_ki = kv_tail != 0 ? 2 : 1;
    for (int qi = 0; qi < n_qi; ++qi)
        for (int ki = 0; ki < n_ki; ++ki)
            CHECK(create_tile_kernels(&kernels.mm1_kernel[qi][ki],
                    &kernels.mm2_kernel[qi][ki], q_rows[qi], kv_w[ki]));

    // Build the JIT online-softmax epilogue: one softmax kernel per
    // (q full/tail, kv full/tail) combination, plus one acc-renormalization
    // kernel per query-block row count.
    using namespace sdpa_softmax_ir;
    // Condition tensor row stride in elements; columns are contiguous. A
    // seq_q axis of extent 1 is a broadcast axis (meaningless stride), so
    // every query row reads the same condition row -> stride 0.
    int cond_stride = 0;
    if (has_select) {
        const memory_desc_wrapper cond_mdw(desc.attn_mask_md());
        if (cond_mdw.dims()[row_dim] != 1)
            cond_stride = static_cast<int>(cond_mdw.strides()[row_dim]);
    }
    for (int qi = 0; qi < n_qi; ++qi) {
        const int m = static_cast<int>(q_rows[qi]);
        for (int ki = 0; ki < n_ki; ++ki) {
            std::unique_ptr<jit_generator_t> k(
                    sdpa_softmax_ir::create_softmax_tile_ir_kernel(isa_for_ir,
                            m, static_cast<int>(kv_w[ki]), has_select,
                            desc.invert_select, cond_stride));
            CHECK(k->create_kernel());
            kernels.softmax_ir_kernel[qi][ki] = std::move(k);
        }
        std::unique_ptr<jit_generator_t> ak(
                sdpa_softmax_ir::create_acc_renorm_ir_kernel(
                        isa_for_ir, m, static_cast<int>(hs_v)));
        CHECK(ak->create_kernel());
        kernels.acc_renorm_ir_kernel[qi] = std::move(ak);
    }

    return status::success;
}

status_t execute(const sdpa_pd_t &pd, const sdpa_online_softmax_conf_t &conf,
        const sdpa_online_softmax_kernels_t &kernels,
        const sdpa_online_softmax_run_args_t &args, void *scratch_base,
        int nthr) {
    constexpr int ndims = sdpa_pd_t::ndims;
    constexpr int row_dim = ndims - 2;

    const sdpa_desc_t &desc = *pd.desc();
    const memory_desc_wrapper q_d(desc.qry_md()), k_d(desc.key_md()),
            v_d(desc.val_md()), out_d(pd.dst_md()), cond_d(desc.attn_mask_md());
    const bool has_select = pd.with_select_mask();

    auto *q_base = static_cast<const char *>(args.q);
    auto *k_base = static_cast<const char *>(args.k);
    auto *v_base = static_cast<const char *>(args.v);
    auto *o_base = static_cast<char *>(args.out);
    const char *cond_base = static_cast<const char *>(args.cond);
    const float scale_val = args.scale;
    const float fill_val = args.fill;

    const dim_t seq_q = desc.queries(), seq_kv = desc.keys(),
                hs_v = desc.values();
    const dim_t kv_blk = conf.kv_blk;
    const dim_t q_blk = conf.q_blk;
    const dim_t group = desc.num_q_heads() / desc.num_kv_heads();
    // Element strides for addressing a KV tile within K / V.
    const dim_t k_col = k_d.strides()[ndims - 1]; // K[.., hs, seq_kv]: seq step
    const dim_t v_row = v_d.strides()[row_dim]; // V[.., seq_kv, hs_v]: kv step
    const dim_t o_row = out_d.strides()[row_dim];
    const dim_t o_col = out_d.strides()[ndims - 1];
    // Broadcast-aware select-condition strides: an axis with extent 1 is a
    // broadcast axis whose stride is meaningless and must contribute 0.
    dims_t eff_cond_strides = {};
    if (has_select)
        for (int d = 0; d < ndims; ++d)
            eff_cond_strides[d]
                    = cond_d.dims()[d] == 1 ? 0 : cond_d.strides()[d];
    const dim_t cond_row = eff_cond_strides[row_dim];
    const dim_t cond_col = eff_cond_strides[ndims - 1];
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    // Q, output, and select-condition tensors use query-head coordinates; K
    // and V use batch/KV-head coordinates shared by each GQA group.
    const auto tensor_base_offset = [](const dims_t &strides, dim_t batch_idx,
                                            dim_t head_idx) -> dim_t {
        return batch_idx * strides[0] + head_idx * strides[1];
    };

    const size_t block_size = conf.scratch_per_thread;
    auto *scratch = static_cast<char *>(scratch_base);

    // Query-axis stride and block count: each work item owns one query block of
    // up to q_blk rows, so the per-thread Q slice, scores tile and running
    // accumulator stay L2-resident across the KV sweep.
    const dim_t q_row = q_d.strides()[row_dim];
    const dim_t n_qblk = utils::div_up(seq_q, q_blk);

    parallel_nd_ext(nthr, desc.batch(), desc.num_q_heads(), n_qblk,
            [&](int tid, int, dim_t bo, dim_t bi, dim_t qblk) {
        const dim_t kvh = bi / group;
        const dim_t q0 = qblk * q_blk;
        const dim_t m = nstl::min(q_blk, seq_q - q0);
        const int qi = m != q_blk ? 1 : 0;

        const float *q_ptr = reinterpret_cast<const float *>(q_base
                + (tensor_base_offset(q_d.strides(), bo, bi) + q0 * q_row)
                        * sizeof(float));
        const float *k_ptr = reinterpret_cast<const float *>(k_base
                + tensor_base_offset(k_d.strides(), bo, kvh) * sizeof(float));
        const float *v_ptr = reinterpret_cast<const float *>(v_base
                + tensor_base_offset(v_d.strides(), bo, kvh) * sizeof(float));
        float *o_ptr = reinterpret_cast<float *>(o_base
                + (tensor_base_offset(out_d.strides(), bo, bi) + q0 * o_row)
                        * sizeof(float));
        const uint8_t *c_ptr = has_select
                ? reinterpret_cast<const uint8_t *>(cond_base
                          + (tensor_base_offset(eff_cond_strides, bo, bi)
                                    + q0 * cond_row)
                                  * sizeof(uint8_t))
                : nullptr;

        // Online-softmax running state kept in a numerically stable form: the
        // accumulator (acc) holds the *normalized* output so far, so its
        // magnitude stays O(|V|). Per tile, mm2 produces the raw P_tile*V_tile
        // into pv, then acc is renormalized. row_max (m) and row_denom (l) are
        // the running max and denominator. Buffers are per-thread slices of
        // the scratchpad (see configure()), sized for one query block (q_blk
        // rows).
        char *tblock = scratch + static_cast<size_t>(tid) * block_size;
        float *scores = reinterpret_cast<float *>(tblock + conf.off_scores);
        float *acc = reinterpret_cast<float *>(tblock + conf.off_acc);
        float *pv = reinterpret_cast<float *>(tblock + conf.off_pv);
        float *row_max = reinterpret_cast<float *>(tblock + conf.off_row_max);
        float *row_denom
                = reinterpret_cast<float *>(tblock + conf.off_row_denom);
        // Per-row renormalization coefficient for the current tile.
        float *old_coef = reinterpret_cast<float *>(tblock + conf.off_old_coef);
        // acc/row_max/row_denom carry running state across tiles, so they must
        // be initialized (scratchpad memory is uninitialized).
        std::fill(row_max, row_max + m, neg_inf);
        std::fill(row_denom, row_denom + m, 0.0f);
        std::fill(acc, acc + static_cast<size_t>(m) * hs_v, 0.0f);

        for (dim_t kv0 = 0; kv0 < seq_kv; kv0 += kv_blk) {
            const dim_t w = nstl::min(kv_blk, seq_kv - kv0);
            const int ki = w != kv_blk ? 1 : 0;
            const auto *mm1 = kernels.mm1_kernel[qi][ki];
            const auto *mm2 = kernels.mm2_kernel[qi][ki];

            // mm1: scores_tile[m, w] = Q_tile[m, hs_qk] * K_tile[hs_qk, w],
            // where Q_tile is this m-row query block and K_tile spans keys
            // [kv0, kv0 + w).
            brgemm_batch_element_t batch1;
            batch1.ptr.A = q_ptr;
            batch1.ptr.B = k_ptr + kv0 * k_col;
            brgemm_kernel_execute(mm1, 1, &batch1, scores, nullptr);

            // Online-softmax epilogue over this KV tile: apply scale + mask,
            // update the running max/denom, and form P_tile = exp(s - m_new).
            const auto &sm = kernels.softmax_ir_kernel[qi][ki];
            sdpa_softmax_ir::softmax_row_args_t sargs;
            sargs.scores = scores;
            sargs.scale = &scale_val;
            sargs.m = row_max;
            sargs.l = row_denom;
            sargs.old_coef = old_coef;
            // cond points at this tile's first column (row 0); the kernel
            // advances by the compiled cond row stride per row.
            sargs.cond = c_ptr ? c_ptr + kv0 * cond_col : nullptr;
            sargs.fill = &fill_val;
            (*sm)(&sargs);

            // mm2: pv_tile[m, hs_v] = P_norm_tile[m, w] * V_tile[w, hs_v],
            // where V_tile spans keys [kv0, kv0 + w).
            brgemm_batch_element_t batch2;
            batch2.ptr.A = scores;
            batch2.ptr.B = v_ptr + kv0 * v_row;
            brgemm_kernel_execute(mm2, 1, &batch2, pv, nullptr);

            // Renormalize the running output: acc = old_coef*acc + pv.
            sdpa_softmax_ir::acc_renorm_args_t aargs;
            aargs.acc = acc;
            aargs.pv = pv;
            aargs.old_coef = old_coef;
            (*kernels.acc_renorm_ir_kernel[qi])(&aargs);
        }

        // acc already holds the normalized output; scatter to user output.
        for (dim_t i = 0; i < m; ++i) {
            const float *arow = acc + i * hs_v;
            float *out_row = o_ptr + i * o_row;
            for (dim_t d = 0; d < hs_v; ++d)
                out_row[d * o_col] = arow[d];
        }
    });

    return status::success;
}

} // namespace sdpa_online_softmax

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
