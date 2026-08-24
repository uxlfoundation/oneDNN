/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
#include "common/broadcast_strategy.hpp"
#include "common/memory_desc_wrapper.hpp"

#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace injector {

namespace {
struct bcast_info_t {
    binary_injector::broadcast_t bcast;
    dim_t C, oc_stride, blk;
};

// Classify the binary rhs vs dst using the shared strategy classifier (x64
// parity), restricted to the strategies the RVV injector implements. per_oc /
// per_oc_spatial / per_w all map to the injector's per-lane gather with
// (C = outer count, oc_stride = outer stride, blk = inner block); scalar and
// no_broadcast map to scalar / per_element.
bcast_info_t classify_binary(
        const memory_desc_t &rhs_md, const memory_desc_t *dst_md) {
    using bt = binary_injector::broadcast_t;
    const memory_desc_wrapper src1_d(rhs_md);
    if (!dst_md)
        return {src1_d.nelems() == 1 ? bt::scalar : bt::per_element, 0, 0, 1};

    const memory_desc_wrapper d(dst_md);
    static const bcast_set_t supported {broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::no_broadcast};
    const auto strat = get_rhs_arg_broadcasting_strategy(rhs_md, d, supported);
    const auto &bd = d.blocking_desc();
    switch (strat) {
        case broadcasting_strategy_t::scalar: return {bt::scalar, 0, 0, 1};
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            // channel = ((o / stride) % (C/blk)) * blk + (o % blk); for a
            // channel-blocked dst (nChw8c) blk is the inner block on dim 1.
            dim_t blk = 1;
            for (int k = 0; k < bd.inner_nblks; k++)
                if (bd.inner_idxs[k] == 1) blk *= bd.inner_blks[k];
            return {bt::per_oc, (d.dims()[1] + blk - 1) / blk, bd.strides[1],
                    blk};
        }
        case broadcasting_strategy_t::per_w: {
            const int wd = d.ndims() - 1;
            return {bt::per_oc, d.dims()[wd], bd.strides[wd], 1};
        }
        default: // no_broadcast (unsupported is gated out by the consumer pd)
            return {bt::per_element, 0, 0, 1};
    }
}
} // namespace

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(
        jit_generator_t *host, const post_ops_t &post_ops,
        const eltwise_injector::static_params_t &eltwise_static_params,
        const binary_injector::static_params_t *binary_static_params,
        const memory_desc_t *dst_md,
        const binary_injector::static_params_t *select_static_params)
    : host_(host), post_ops_(post_ops) {
    eltwise_injectors_.reserve(post_ops.len());
    binary_injectors_.reserve(post_ops.len());
    int arg_idx = 0; // this binary's slot in the rhs pointer array
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &e = post_ops.entry_[i];
        if (e.is_eltwise()) {
            eltwise_injectors_.emplace_back(
                    host_, e.eltwise, eltwise_static_params);
        } else if (e.is_binary()) {
            assert(binary_static_params != nullptr
                    && "binary post-op requires binary scratch");
            const bcast_info_t info
                    = classify_binary(e.binary.src1_desc, dst_md);
            // per-binary static params: strategy + channel map + rhs dtype.
            binary_injector::static_params_t bsp = *binary_static_params;
            bsp.bcast = info.bcast;
            bsp.C = info.C;
            bsp.oc_stride = info.oc_stride;
            bsp.blk = info.blk;
            bsp.rhs_dt = e.binary.src1_desc.data_type;
            if (e.is_binary_with_ternary_op()) {
                assert(select_static_params != nullptr
                        && "binary select requires condition scratch");
                const bcast_info_t select_info
                        = classify_binary(e.binary.src2_desc, dst_md);
                binary_injector::static_params_t ssp = *select_static_params;
                ssp.bcast = select_info.bcast;
                ssp.C = select_info.C;
                ssp.oc_stride = select_info.oc_stride;
                ssp.blk = select_info.blk;
                ssp.rhs_dt = e.binary.src2_desc.data_type;
                binary_injectors_.emplace_back(host_, e.binary.alg, info.bcast,
                        bsp, arg_idx, select_info.bcast, &ssp);
                arg_idx += 2;
            } else {
                binary_injectors_.emplace_back(
                        host_, e.binary.alg, info.bcast, bsp, arg_idx++);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_body(const Vmm &v,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
        int entry_begin, int entry_end) {
    // Apply entries [entry_begin, entry_end) in attribute order: eltwise and
    // binary injectors are consumed in creation order (binaries index the rhs
    // array by the arg_idx baked in at construction). Non-eltwise/non-binary
    // entries (sum) are skipped — the host applies sum at its position. Advance
    // the injector counters over the skipped prefix so the sub-range picks the
    // right injectors.
    size_t e_idx = 0, b_idx = 0;
    for (int i = 0; i < entry_begin; i++) {
        if (post_ops_.entry_[i].is_eltwise())
            e_idx++;
        else if (post_ops_.entry_[i].is_binary())
            b_idx++;
    }
    for (int i = entry_begin; i < entry_end; i++) {
        const auto &e = post_ops_.entry_[i];
        if (e.is_eltwise())
            eltwise_injectors_[e_idx++].compute_vector(v.getIdx());
        else if (e.is_binary())
            binary_injectors_[b_idx++].compute_vector(
                    v.getIdx(), rhs_arg_params);
    }
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {
    // NB: a per-element/per_oc binary post-op reads a per-register rhs slice, so
    // a multi-register range is only correct when every register's offset is in
    // rhs_arg_params; all current consumers pass exactly one register (an
    // eltwise-only chain is safe for any range).
    for (size_t i = start_idx; i < end_idx; i++)
        compute_body(Vmm(i), rhs_arg_params, 0, (int)post_ops_.len());
}

template <cpu_isa_t isa>
bool jit_uni_postops_injector_t<isa>::post_ops_ok(
        const post_ops_t &post_ops, int n_vaux, bool allow_binary_select) {
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &e = post_ops.entry_[i];
        if (e.is_eltwise()) {
            const auto alg = e.eltwise.alg;
            const bool ok = eltwise_injector::is_alg_supported(alg)
                    || (n_vaux >= 4 && eltwise_injector::needs_extra_aux(alg));
            if (!ok) return false;
        } else if (e.is_binary()) {
            if (e.binary.alg == alg_kind::binary_select) {
                if (!allow_binary_select) return false;
            } else if (!binary_injector::is_alg_supported(e.binary.alg))
                return false;
        } else {
            return false; // sum/prelu/... -> consumer falls back to a ref impl
        }
    }
    return true;
}

template struct jit_uni_postops_injector_t<v>;
template struct jit_uni_postops_injector_t<zvfh>;
template struct jit_uni_postops_injector_t<zvfbfwma>;

} // namespace injector
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
