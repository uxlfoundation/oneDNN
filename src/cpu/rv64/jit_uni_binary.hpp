/*******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_JIT_UNI_BINARY_HPP
#define CPU_RV64_JIT_UNI_BINARY_HPP

#include <cstdint>
#include <limits>
#include <memory>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_binary_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_uni_binary_kernel_t;

// Standalone binary primitive: a VLA JIT wrapper that computes
// (scale0*src0) OP (scale1*src1) in f32, applies the sum + eltwise + binary
// post-op chain, and stores to dst (converted/saturated at the boundary),
// mirroring x64/aarch64 jit_uni_binary_t. Supports:
//   - dtypes f32/f16/s32/s8/u8, freely mixed across src0/src1/dst (f16 needs zvfh)
//   - arbitrary src1 broadcast over plain (nchw/nhwc/...) and single-inner-block
//     (nChw4c/8c/16c, including a padded channel tail) dst, plus src0/src1
//     different plain layouts (nchw:nhwc, read via a strided src1 load)
//   - per-tensor src0/src1 scales
//   - a post-op chain: same-parameter sums at any positions (applied in-kernel)
//     plus any number of eltwise ops (incl. log/soft_relu/gelu_erf, which fit the
//     available aux budget) and binary ops (any supported rhs dtype; scalar /
//     per_element / per_oc / per_oc_spatial / per_w broadcast)
//   - ternary select (dst = src2 ? src0 : src1) folded into the same f32 path,
//     so it also supports broadcast, scales, and post-ops; src2 is s8
// A single instance handles every case internally (RVV has one vector isa).
struct jit_uni_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;
        DECLARE_COMMON_PD_T("jit:uni", jit_uni_binary_t);

        status_t init(const engine_t *engine) {
            UNUSED(engine);
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;
            const data_type_t dd = dst_md()->data_type;
            const data_type_t d0 = src_md(0)->data_type;
            const data_type_t d1 = src_md(1)->data_type;

            // Pure JIT, registered via CPU_INSTANCE_RV64 (runtime dispatch):
            // gate on the V extension, and on zvfh if any f16 operand is present.
            VDISPATCH_BINARY(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
            auto dt_ok = [](data_type_t dt) {
                return utils::one_of(dt, f32, f16, s32, s8, u8);
            };
            VDISPATCH_BINARY(dt_ok(dd) && dt_ok(d0) && dt_ok(d1),
                    VERBOSE_UNSUPPORTED_DT);
            const bool any_f16 = utils::one_of(f16, dd, d0, d1);
            VDISPATCH_BINARY(IMPLICATION(any_f16, mayiuse(zvfh)),
                    VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_BINARY(platform::has_data_type_support(dd)
                            && platform::has_data_type_support(d0)
                            && platform::has_data_type_support(d1),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BINARY(check_alg(), VERBOSE_BAD_ALGORITHM);

            // Attributes: per-tensor src0/src1 scales + a post-op chain.
            VDISPATCH_BINARY(
                    attr()->has_default_values(sm::post_ops | sm::scales),
                    VERBOSE_UNSUPPORTED_ATTR);
            const bool scales_ok
                    = attr_scales_ok({DNNL_ARG_SRC_0, DNNL_ARG_SRC_1});
            VDISPATCH_BINARY(scales_ok, VERBOSE_UNSUPPORTED_SCALES_CFG);
            // Resolve formats before classifying the post-op chain: the post-op
            // binary rhs (and dst) may arrive as `any`, and post_ops_supported()
            // compares the rhs layout to dst — mirror x64's ordering so a
            // dst-shaped (per_element) rhs is not spuriously rejected.
            VDISPATCH_BINARY_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY_SC(attr_.set_default_formats(dst_md()),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY(post_ops_supported(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_BINARY(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            const bool is_select = desc()->alg_kind == alg_kind::binary_select;
            if (is_select) {
                // Ternary select is folded into the general f32 path (broadcast,
                // scales, sum + post-ops, mixed src0/src1/dst dtypes). The common
                // binary descriptor forces the condition src2 to s8; it must be
                // full (dst-shaped, no broadcast — read contiguously per run).
                VDISPATCH_BINARY(
                        src_md(2)->data_type == s8, VERBOSE_UNSUPPORTED_DT);
                const memory_desc_wrapper s2_d(src_md(2));
                VDISPATCH_BINARY(s2_d.similar_to(memory_desc_wrapper(dst_md()),
                                         true, false),
                        VERBOSE_UNSUPPORTED_TAG);
            }

            VDISPATCH_BINARY(init_conf(), VERBOSE_UNSUPPORTED_TAG);
            return status::success;
        }

        // --- broadcast / iteration plan (filled by init_conf) ---
        // whole_: one flat pass over the tensor (scalar or no-broadcast). Else
        // the driver iterates n_outer_ runs of inner_ elements, offsetting src1
        // by the per-dim broadcast strides s1_str_.
        bool whole_ = false;
        bool scalar_whole_ = false; // whole_ && src1 is a single scalar
        bool scalar_inner_ = false; // per-run: src1 broadcasts the inner dim
        // src1 inner-dim element stride: 1 = contiguous, >1 = strided (src0/src1
        // different plain layouts, e.g. nchw:nhwc); 0/unused when scalar_inner_.
        dim_t s1_inner_stride_ = 1;
        int nd_ = 0;
        dim_t inner_ = 0; // elements per run (dst last dim)
        dim_t n_outer_ = 0; // number of runs
        dim_t total_ = 0; // dst element count
        // A padded single channel block is processed as one run per physical
        // block. The last C block uses tail_ active lanes and its remaining dst
        // padding is explicitly zeroed by the driver.
        dim_t tail_ = 0;
        int tail_axis_ = -1; // physical outer axis containing the C-block index
        bool s1_same_layout_
                = false; // src1 offset follows the physical dst run
        dims_t out_dims_ = {}; // dst dims [0..nd-2] (run-major decomposition)
        dims_t s1_str_ = {}; // src1 element stride per dst dim (0 == broadcast)

        // --- attribute plan ---
        bool do_scale0_ = false, do_scale1_ = false;
        bool do_sum_ = false;
        float sum_scale_ = 0.f;
        post_ops_t po_; // full post-op chain (kernel applies each sum in place)

        bool check_alg() const {
            using namespace alg_kind;
            return utils::one_of(desc()->alg_kind, binary_add, binary_sub,
                    binary_mul, binary_div, binary_max, binary_min,
                    binary_select, binary_ge, binary_gt, binary_le, binary_lt,
                    binary_eq, binary_ne);
        }

        // Post-ops: sums applied by the kernel at their chain positions (x64/
        // AArch64 parity: multiple sums are allowed when scale/zero-point match),
        // plus a chain of eltwise / binary ops the injector can emit.
        bool post_ops_supported() const {
            const auto &po = attr()->post_ops_;
            const int sum_idx = po.find(primitive_kind::sum);
            if (sum_idx != -1) {
                const auto &first = po.entry_[sum_idx].sum;
                for (int i = sum_idx; i < po.len(); i++) {
                    if (!po.entry_[i].is_sum(false, false)) continue;
                    const auto &s = po.entry_[i].sum;
                    if (s.dt != data_type::undef && s.dt != dst_md()->data_type)
                        return false;
                    if (s.zero_point != 0 || s.scale != first.scale
                            || s.zero_point != first.zero_point)
                        return false;
                }
            }
            for (int i = 0; i < po.len(); i++) {
                const auto &e = po.entry_[i];
                if (e.is_sum(false, false)) continue; // handled in-kernel
                if (e.is_eltwise()) {
                    // the kernel supplies 4 aux, so the heavy eltwise algs
                    // (log/soft_relu/gelu_erf) are available here too.
                    const auto a = e.eltwise.alg;
                    if (!eltwise_injector::is_alg_supported(a)
                            && !eltwise_injector::needs_extra_aux(a))
                        return false;
                } else if (e.is_binary()) {
                    if (e.binary.alg != alg_kind::binary_select
                            && !binary_injector::is_alg_supported(e.binary.alg))
                        return false;
                    static const bcast_set_t sb {
                            broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_oc,
                            broadcasting_strategy_t::per_oc_spatial,
                            broadcasting_strategy_t::per_w,
                            broadcasting_strategy_t::no_broadcast};
                    const memory_desc_wrapper dw(dst_md());
                    // Each operand is converted to f32 at load. Select validates
                    // its independent condition descriptor with the same current
                    // dtype/broadcast/layout contract as src1.
                    const auto rhs_ok = [&](const memory_desc_t &md) {
                        const memory_desc_wrapper rhs_d(md);
                        const data_type_t rdt = md.data_type;
                        if (!utils::one_of(rdt, data_type::f32, data_type::f16,
                                    data_type::s32, data_type::s8,
                                    data_type::u8))
                            return false;
                        if (rdt == data_type::f16 && !mayiuse(zvfh))
                            return false;
                        // The RVV address paths consume a packed logical rhs.
                        // Explicit descriptors with holes must fall through,
                        // matching the x64/AArch64 injector applicability gate.
                        if (!rhs_d.is_dense(true)) return false;
                        const auto strat
                                = get_rhs_arg_broadcasting_strategy(md, dw, sb);
                        if (strat == broadcasting_strategy_t::unsupported)
                            return false;
                        if (strat == broadcasting_strategy_t::no_broadcast
                                && !memory_desc_wrapper(md).similar_to(
                                        dw, true, false))
                            return false;
                        const bool indexed = utils::one_of(strat,
                                broadcasting_strategy_t::per_oc,
                                broadcasting_strategy_t::per_oc_spatial,
                                broadcasting_strategy_t::per_w);
                        if (indexed) {
                            // per_oc/per_w uses an e32 lane index and vluxei32
                            // byte offsets. Reject shapes whose flattened output
                            // position, divisor, stride, or rhs byte offset cannot
                            // be represented without wrapping.
                            constexpr uint64_t max_u32
                                    = std::numeric_limits<uint32_t>::max();
                            const auto last_index_fits = [&](dim_t nelems) {
                                return nelems <= 0
                                        || static_cast<uint64_t>(nelems - 1)
                                        <= max_u32;
                            };
                            if (!last_index_fits(dw.nelems(true))) return false;
                            const dim_t rhs_nelems = rhs_d.nelems(true);
                            const size_t rhs_esz = rhs_d.data_type_size();
                            if (rhs_nelems > 0
                                    && static_cast<uint64_t>(rhs_nelems - 1)
                                            > max_u32 / rhs_esz)
                                return false;

                            const auto &bd = dw.blocking_desc();
                            dim_t count = 0;
                            dim_t stride = 0;
                            dim_t blk = 1;
                            if (strat == broadcasting_strategy_t::per_w) {
                                const int wd = dw.ndims() - 1;
                                count = dw.dims()[wd];
                                stride = bd.strides[wd];
                            } else {
                                for (int k = 0; k < bd.inner_nblks; k++)
                                    if (bd.inner_idxs[k] == 1)
                                        blk *= bd.inner_blks[k];
                                if (blk > 16 || (blk & (blk - 1)) != 0)
                                    return false;
                                count = (dw.dims()[1] + blk - 1) / blk;
                                stride = bd.strides[1];
                            }
                            if (count <= 0 || stride <= 0
                                    || static_cast<uint64_t>(count) > max_u32
                                    || static_cast<uint64_t>(stride) > max_u32)
                                return false;
                        }
                        return true;
                    };
                    if (!rhs_ok(e.binary.src1_desc)) return false;
                    if (e.is_binary_with_ternary_op()
                            && !rhs_ok(e.binary.src2_desc))
                        return false;
                } else
                    return false; // prelu / etc -> ref
            }
            return true;
        }

    private:
        // Layout equality ignoring data type (dims + strides + blocking).
        static bool layout_eq(
                const memory_desc_wrapper &a, const memory_desc_wrapper &b) {
            return a.similar_to(b, true, false);
        }

        bool init_conf() {
            const memory_desc_wrapper s0(src_md(0));
            const memory_desc_wrapper s1(src_md(1));
            const memory_desc_wrapper d(dst_md());
            if (!d.is_dense(true) || !s0.is_dense(true) || !s1.is_dense(true))
                return false;
            // src0 must match dst layout exactly (no src0 broadcast).
            if (!layout_eq(s0, d)) return false;

            nd_ = d.ndims();
            total_ = d.nelems(false);
            const bool has_padding = d.nelems(true) != d.nelems(false);

            // scales
            do_scale0_ = !attr()->scales_.has_default_values(DNNL_ARG_SRC_0);
            do_scale1_ = !attr()->scales_.has_default_values(DNNL_ARG_SRC_1);

            // Full post-op chain; the kernel applies every sum at its position
            // and the injector skips those entries.
            const auto &po = attr()->post_ops_;
            const int sum_pos = po.find(primitive_kind::sum);
            do_sum_ = sum_pos != -1;
            sum_scale_ = do_sum_ ? po.entry_[sum_pos].sum.scale : 0.f;
            po_ = po;

            // src1 whole-tensor cases
            if (!has_padding && s1.nelems(false) == 1) {
                whole_ = true;
                scalar_whole_ = true;
                return true;
            }
            if (!has_padding && layout_eq(s1, d)) {
                whole_ = true;
                scalar_whole_ = false;
                return true;
            }

            // General broadcast over a plain OR single-inner-block dst (nchw,
            // nhwc, nChw8c/16c, ...). src1 may be plain or use the same inner
            // block as dst; dst may carry at most one inner block.
            const auto &dbd = d.blocking_desc();
            const auto &s1bd = s1.blocking_desc();
            s1_same_layout_ = layout_eq(s1, d);
            const bool scalar_s1 = s1.nelems(false) == 1;
            if (dbd.inner_nblks > 1) return false; // at most one inner block

            const int bdim = dbd.inner_nblks == 1 ? dbd.inner_idxs[0] : -1;
            const dim_t blk = dbd.inner_nblks == 1 ? dbd.inner_blks[0] : 1;
            const bool s1_blocked = s1bd.inner_nblks == 1 && bdim >= 0
                    && s1bd.inner_idxs[0] == bdim && s1bd.inner_blks[0] == blk;
            if (s1bd.inner_nblks != 0 && !s1_blocked && !scalar_s1)
                return false;

            const dim_t *dd = d.dims();
            const dim_t *d1 = s1.dims();
            const auto &dstr = dbd.strides;
            const auto &s1str = s1bd.strides;
            for (int i = 0; i < nd_; i++)
                if (d1[i] != 1 && d1[i] != dd[i]) return false;

            // Build the physical dims outer->inner: each logical dim's outer
            // part (sorted by dst stride, memory order), then the inner block
            // (if any) as the unit-stride innermost. For the blocked dim, the
            // outer part advances src1 by blk channels per step. The run index
            // then maps to contiguous dst blocks and the correct src1 offset.
            if (has_padding) {
                // Match the x64/AArch64 supported tail family: one power-of-two
                // channel block no wider than 16, with padding only on C.
                if (bdim != 1 || blk > 16 || (blk & (blk - 1)) != 0)
                    return false;
                for (int i = 0; i < nd_; i++)
                    if (i != bdim && d.padded_dims()[i] != dd[i]) return false;
                if (d.padded_dims()[bdim] != ((dd[bdim] + blk - 1) / blk) * blk)
                    return false;
                tail_ = dd[bdim] % blk;
            }
            struct phys_t {
                dim_t size, s1s, sort;
                int logical_dim;
            } ph[DNNL_MAX_NDIMS + 1];
            int np = 0;
            for (int i = 0; i < nd_; i++) {
                const dim_t osz = (i == bdim) ? (dd[i] + blk - 1) / blk : dd[i];
                if (osz < 1) return false;
                const dim_t s1s = d1[i] == 1
                        ? 0
                        : (i == bdim && !s1_blocked ? s1str[i] * blk
                                                    : s1str[i]);
                ph[np++] = {osz, s1s, dstr[i], i};
            }
            for (int a = 0; a < np; a++)
                for (int b = a + 1; b < np; b++)
                    if (ph[b].sort > ph[a].sort) {
                        const phys_t t = ph[a];
                        ph[a] = ph[b];
                        ph[b] = t;
                    }
            if (bdim >= 0) {
                const dim_t s1s_in
                        = d1[bdim] == 1 ? 0 : (s1_blocked ? 1 : s1str[bdim]);
                ph[np++] = {blk, s1s_in, 1, bdim};
            }

            // Coalesce trailing physical dims into one longer run while the
            // dst stays contiguous across the boundary and src1 advances
            // uniformly (both dims broadcast, or the outer stride continues
            // the inner element pattern). Short innermost runs otherwise
            // dominate with per-call overhead: channels-first per-C measured
            // ~2x slower with W-sized runs than with plane-sized runs. Padded
            // tensors keep per-block runs (the zeroed tail must stay a run
            // suffix); keep at least nthr runs so the driver can parallelize.
            const dim_t min_outer = dnnl_get_max_threads();
            while (tail_ == 0 && np >= 2) {
                const phys_t in = ph[np - 1];
                const phys_t out = ph[np - 2];
                if (out.sort != in.sort * in.size) break;
                const bool both_bcast = out.s1s == 0 && in.s1s == 0;
                const bool uniform = in.s1s != 0 && out.s1s == in.s1s * in.size;
                if (!both_bcast && !uniform) break;
                dim_t outer = 1;
                for (int k = 0; k + 2 < np; k++)
                    outer *= ph[k].size;
                if (outer < min_outer) break;
                ph[np - 2].size = out.size * in.size;
                ph[np - 2].s1s = in.s1s;
                ph[np - 2].sort = in.sort;
                np--;
            }

            const phys_t &in = ph[np - 1];
            if (in.sort != 1) return false; // innermost must be dst-unit-stride
            inner_ = in.size;
            scalar_inner_ = in.s1s == 0;
            // src1 inner stride: 0 = broadcast (scalar), 1 = contiguous (vle),
            // >1 = strided (vlse) — src0/src1 different plain layouts, e.g.
            // nchw:nhwc, mirroring x64's is_different_layouts_allowed.
            s1_inner_stride_ = in.s1s;
            n_outer_ = 1;
            nd_ = np; // physical dim count drives the execute decomposition
            for (int k = 0; k < np - 1; k++) {
                out_dims_[k] = ph[k].size;
                s1_str_[k] = ph[k].s1s;
                n_outer_ *= ph[k].size;
                if (tail_ != 0 && ph[k].logical_dim == bdim) tail_axis_ = k;
            }
            if (tail_ != 0 && tail_axis_ < 0) return false;
            whole_ = false;
            return true;
        }
    };

    jit_uni_binary_t(const pd_t *apd);
    ~jit_uni_binary_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_binary_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_BINARY_HPP
