/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/rv64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace binary_injector {

using namespace Xbyak_riscv;

// Fixed scalar scratch, the aarch64 X_TMP_0..4 analogs: x64 likewise clobbers
// fixed helper GPRs (rax/rdx/r8/r9) and shields them with the register-
// preserve guard when the host asks for preservation. Hosts must not pick
// these as rhs_addr_reg/rhs_helper_reg/rhs_addr_cache_reg.
static const Reg X_TMP_0 = t4;
static const Reg X_TMP_1 = t5;
static const Reg X_TMP_2 = t6;
static const Reg X_TMP_3 = a6;
static const Reg X_TMP_4 = a7;

#define VCHECK_BIN_INJ_BOOL(cond, msg) \
    VCONDCHECK(primitive, create, check, binary_injector, cond, false, msg);

// The strategy set the RVV injector realizes -- the full aarch64 set. Every
// non-scalar/non-contiguous strategy (per_oc, per_oc_spatial, per_w,
// per_mb_spatial, per_mb_w) is realized via the RVV per-lane gather (each lane
// resolves its own rhs index; a VLA run spans the broadcast dim). per_mb_spatial
// and per_mb_w are restricted to plain ncsp/nspc dst layouts (see
// is_bcast_supported); their blocked/cspn variants are not wired. As on
// x64/aarch64 this is the injector's ability set: each host primitive
// advertises its own subset (the binary primitive uses the x64 set, which
// excludes per_mb_spatial/per_mb_w).
static bcast_set_t get_all_strategies_supported_by_injector() {
    return bcast_set_t {broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::per_mb_spatial,
            broadcasting_strategy_t::per_mb_w, broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::no_broadcast};
}

bool is_data_supported(cpu_isa_t isa, data_type_t data_type) {
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
        case data_type::s8:
        case data_type::u8: return true;
        case data_type::bf16: return (isa & zvfbfwma) == zvfbfwma;
        case data_type::f16: return (isa & zvfh) == zvfh;
        default: return false;
    }
}

bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    VCHECK_BIN_INJ_BOOL(is_data_supported(isa, src1_desc.data_type),
            VERBOSE_ISA_DT_MISMATCH);

    VCHECK_BIN_INJ_BOOL(memory_desc_wrapper(src1_desc).is_dense(true),
            VERBOSE_NONTRIVIAL_STRIDE);

    return is_bcast_supported(src1_desc, dst_d, supported_strategy_set);
}

static bool src1_desc_layout_same_as_dst_d(
        const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d) {
    if (dst_d.md_ == nullptr) return false;
    const auto &lhs = src1_desc;
    const auto &rhs = *(dst_d.md_);

    using namespace dnnl::impl::utils;
    const bool is_format_any
            = one_of(format_kind::any, lhs.format_kind, rhs.format_kind);

    return lhs.ndims == rhs.ndims
            && (is_format_any
                    || (lhs.format_kind == rhs.format_kind
                            && array_cmp(lhs.format_desc.blocking.strides,
                                    rhs.format_desc.blocking.strides,
                                    lhs.ndims)))
            && array_cmp(lhs.dims, rhs.dims, lhs.ndims)
            && array_cmp(lhs.padded_dims, rhs.padded_dims, lhs.ndims)
            && array_cmp(lhs.padded_offsets, rhs.padded_offsets, lhs.ndims)
            && lhs.offset0 == rhs.offset0;
}

bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
            src1_desc, dst_d, supported_strategy_set);

    VCHECK_BIN_INJ_BOOL(
            IMPLICATION(bcast_type == broadcasting_strategy_t::no_broadcast,
                    src1_desc_layout_same_as_dst_d(src1_desc, dst_d)),
            "src1 and dst must have the same layout if not broadcasting");

    if (bcast_type == broadcasting_strategy_t::no_broadcast) return true;

    VCHECK_BIN_INJ_BOOL(bcast_type != broadcasting_strategy_t::unsupported,
            "Unsupported broadcast type");

    // The per_oc/per_oc_spatial/per_w/per_mb_spatial/per_mb_w strategies read the
    // rhs through a per-lane gather (vluxei32, e32 lane index, blocked-channel
    // low bits via vand_vi's 5-bit immediate). Reject shapes whose flattened
    // output position, divisor, stride, or rhs byte offset cannot be represented
    // without wrapping.
    if (utils::one_of(bcast_type, broadcasting_strategy_t::per_oc,
                broadcasting_strategy_t::per_oc_spatial,
                broadcasting_strategy_t::per_w,
                broadcasting_strategy_t::per_mb_spatial,
                broadcasting_strategy_t::per_mb_w)) {
        constexpr uint64_t max_u32 = std::numeric_limits<uint32_t>::max();
        const memory_desc_wrapper rhs_d(src1_desc);
        const dim_t dst_nelems = dst_d.nelems(true);
        VCHECK_BIN_INJ_BOOL(dst_nelems <= 0
                        || static_cast<uint64_t>(dst_nelems - 1) <= max_u32,
                "Gather index does not fit into 32 bits");
        const dim_t rhs_nelems = rhs_d.nelems(true);
        const size_t rhs_esz = rhs_d.data_type_size();
        VCHECK_BIN_INJ_BOOL(rhs_nelems <= 0
                        || static_cast<uint64_t>(rhs_nelems - 1)
                                <= max_u32 / rhs_esz,
                "Gather byte offset does not fit into 32 bits");
        // per_oc/per_oc_spatial extract the blocked-channel low bits with
        // vand_vi (5-bit immediate), so their channel inner-block must be a
        // power of two <= 16. per_w computes the block with vdivu (any block).
        if (utils::one_of(bcast_type, broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial)) {
            const auto &bd = dst_d.blocking_desc();
            dim_t blk = 1;
            for (int k = 0; k < bd.inner_nblks; k++)
                if (bd.inner_idxs[k] == 1) blk *= bd.inner_blks[k];
            VCHECK_BIN_INJ_BOOL(blk <= 16 && (blk & (blk - 1)) == 0,
                    "Unsupported channel inner block for gather");
        }
        // per_mb_spatial/per_mb_w gather the rhs by its plain logical index
        // (rhs_idx = n*sp_size + spatial), which equals the rhs physical offset
        // only when the rhs is plain. For a blocked/cspn dst the `any` post-op
        // rhs is resolved to a matching non-plain layout whose physical offset
        // differs (by the block/transpose structure) -> route those to the
        // reference; only plain ncsp/nspc are wired.
        if (utils::one_of(bcast_type, broadcasting_strategy_t::per_mb_spatial,
                    broadcasting_strategy_t::per_mb_w)) {
            const auto layout = injector_utils::get_layout_type(dst_d);
            VCHECK_BIN_INJ_BOOL(
                    utils::one_of(layout, injector_utils::layout_t::ncsp,
                            injector_utils::layout_t::nspc),
                    "per_mb_spatial/per_mb_w gather needs a plain ncsp/nspc "
                    "dst");
        }
    }
    return true;
}

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    entry.binary.src1_desc, dst_d, supported_strategy_set);
            return bcast_type == broadcasting_strategy_t::unsupported;
        }
        return false;
    });
}

bool any_binary_postop_rhs_non_scalar_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_like_binary()) {
            const auto bcast_type
                    = get_rhs_arg_broadcasting_strategy(entry.binary.src1_desc,
                            dst_d, get_all_strategies_supported_by_injector());
            return !utils::one_of(bcast_type, broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::unsupported);
        }
        return false;
    });
}

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) {
        if (entry.is_binary()) {
            const memory_desc_wrapper rhs_arg_d(entry.binary.src1_desc);
            return rhs_arg_d.matches_tag(tag);
        }
        return true;
    });
}

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return any_binary_postop_rhs_per_oc_broadcast(
            post_ops, dst_d, get_all_strategies_supported_by_injector());
}

bool any_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    entry.binary.src1_desc, dst_d, supported_strategy_set);
            return bcast_type == broadcasting_strategy_t::per_oc
                    || bcast_type == broadcasting_strategy_t::per_oc_spatial;
        }
        return false;
    });
}

static_params_t::static_params_t(const Xbyak_riscv::Reg &param1,
        const bcast_set_t &supported_strategy_set,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : param1(param1)
    , supported_strategy_set(supported_strategy_set)
    , rhs_arg_static_params(rhs_arg_static_params) {}

static_params_t::static_params_t(const Xbyak_riscv::Reg &param1,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : static_params_t(param1, get_all_strategies_supported_by_injector(),
              rhs_arg_static_params) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak_riscv::Reg &rhs_addr_reg,
        const Xbyak_riscv::Reg &rhs_helper_reg,
        const Xbyak_riscv::Reg &rhs_addr_cache_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, 0, dst_d,
              false /*is_dst_orig_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak_riscv::Reg &rhs_addr_reg,
        const Xbyak_riscv::Reg &rhs_helper_reg,
        const Xbyak_riscv::Reg &rhs_addr_cache_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        std::size_t dst_orig_offset, const memory_desc_wrapper &dst_d)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, dst_orig_offset, dst_d,
              true /*is_dst_orig_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak_riscv::Reg &rhs_addr_reg,
        const Xbyak_riscv::Reg &rhs_helper_reg,
        const Xbyak_riscv::Reg &rhs_addr_cache_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        std::size_t dst_orig_offset, const memory_desc_wrapper &dst_d,
        bool is_dst_orig_set)
    : rhs_dt_helper_vmm_idx(rhs_dt_helper_vmm_idx)
    , rhs_addr_reg(rhs_addr_reg)
    , rhs_helper_reg(rhs_helper_reg)
    , rhs_addr_cache_reg(rhs_addr_cache_reg)
    , preserve_gpr_helpers(preserve_gpr_helpers)
    , preserve_vmm_helper(preserve_vmm_helper)
    , abi_param_offset(abi_param_offset)
    , dst_orig_offset(dst_orig_offset)
    , dst_d(dst_d)
    , is_dst_orig_set_(is_dst_orig_set) {
    // Only the LMUL-independent conditions can be checked here: the helper is a
    // group of group_stride registers, but group_stride is a per-call property
    // of compute_vector_range(), not of the static params. Hardcoding a
    // four-register alignment here would wrongly reject a legal m1/m2 host
    // (e.g. helper v5 at m1). Alignment, range and overlap against the
    // processed accumulators are validated per call by validate_temp_vmm_hint()
    // with JIT_ASSERT, so a Release build fails kernel creation rather than
    // emitting corrupted code. v0 is excluded here because it is the
    // architectural mask register at every LMUL.
    assert(rhs_dt_helper_vmm_idx != 0
            && rhs_dt_helper_vmm_idx < jit_isa_traits_t<v>::n_vregs);
}

template <cpu_isa_t isa>
jit_uni_binary_injector_t<isa>::jit_uni_binary_injector_t(
        jit_generator_t *host, const static_params_t &static_params)
    : host_(host)
    , rhs_arg_static_params_(static_params.rhs_arg_static_params)
    , param1_(static_params.param1)
    , supported_strategy_set_(static_params.supported_strategy_set) {}

template <typename ParamsMap>
static bool params_differ(ParamsMap &params,
        const typename ParamsMap::key_type key1,
        const typename ParamsMap::key_type key2) {
    const auto &it1 = params.find(key1);
    const auto &it2 = params.find(key2);
    if (utils::one_of(params.end(), it1, it2)) return it1 != it2;
    return it1->second != it2->second;
}

static bool params_differ_reg(const std::map<int, Xbyak_riscv::Reg> &params,
        const std::map<int, Xbyak_riscv::Reg>::key_type key1,
        const std::map<int, Xbyak_riscv::Reg>::key_type key2) {
    const auto &it1 = params.find(key1);
    const auto &it2 = params.find(key2);
    if (utils::one_of(params.end(), it1, it2)) return it1 != it2;
    return it1->second.getIdx() != it2->second.getIdx();
}

// Host-positioned mode (rv64-only; A-class): pool/gnorm/conv-epilogue compute
// the rhs offset themselves and construct the injector without a dst descriptor
// (memory_desc_wrapper substitutes the global zero md for a null pointer, so the
// mode is detected via is_zero()). The strategy then reduces to scalar vs
// no_broadcast, derived from the rhs element count; the host supplies the byte
// offset (off_is_bytes) and optional stride (is_strided).
static broadcasting_strategy_t get_rhs_arg_broadcasting_strategy_or_hosted(
        const memory_desc_t &rhs_md, const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    if (dst_d.is_zero())
        return memory_desc_wrapper(rhs_md).nelems() == 1
                ? broadcasting_strategy_t::scalar
                : broadcasting_strategy_t::no_broadcast;
    return get_rhs_arg_broadcasting_strategy(
            rhs_md, dst_d, supported_strategy_set);
}

static bool rhs_arg_params_differ(size_t vmm_idx1, size_t vmm_idx2,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        broadcasting_strategy_t rhs_broadcasting_strategy) {

    const auto &out_addr = rhs_arg_params.vmm_idx_to_out_addr;
    const auto &out_reg = rhs_arg_params.vmm_idx_to_out_reg;

    const auto &out_elem_off_addr = rhs_arg_params.vmm_idx_to_out_elem_off_addr;
    const auto &out_elem_off_val = rhs_arg_params.vmm_idx_to_out_elem_off_val;
    const auto &out_off_oprnd = rhs_arg_params.vmm_idx_to_out_off_oprnd;
    const auto &oc_off_addr = rhs_arg_params.vmm_idx_to_oc_elem_off_addr;
    const auto &oc_off_val = rhs_arg_params.vmm_idx_to_oc_elem_off_val;
    const auto &oc_off_oprnd = rhs_arg_params.vmm_idx_to_oc_off_oprnd;
    const auto &sp_off_addr = rhs_arg_params.vmm_idx_to_sp_elem_off_addr;
    const auto &sp_off_val = rhs_arg_params.vmm_idx_to_sp_elem_off_val;
    const auto &sp_off_oprnd = rhs_arg_params.vmm_idx_to_sp_off_oprnd;

    if (rhs_broadcasting_strategy == broadcasting_strategy_t::scalar) {
        return false;
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::no_broadcast) {
        return params_differ(out_addr, vmm_idx1, vmm_idx2)
                || params_differ_reg(out_reg, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(out_off_oprnd, vmm_idx1, vmm_idx2);
    } else if (rhs_broadcasting_strategy == broadcasting_strategy_t::per_oc
            || rhs_broadcasting_strategy
                    == broadcasting_strategy_t::per_oc_spatial) {
        return params_differ(out_addr, vmm_idx1, vmm_idx2)
                || params_differ_reg(out_reg, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_val, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_oprnd, vmm_idx1, vmm_idx2);
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::per_mb_spatial) {
        return params_differ(out_addr, vmm_idx1, vmm_idx2)
                || params_differ_reg(out_reg, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(sp_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(sp_off_val, vmm_idx1, vmm_idx2)
                || params_differ(sp_off_oprnd, vmm_idx1, vmm_idx2);
    }
    return true;
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::validate_rhs_scratch(
        const injector_utils::vmm_index_set_t &vmm_idxs, size_t group_stride,
        const dnnl_post_ops::entry_t &post_op,
        broadcasting_strategy_t rhs_broadcasting_strategy) const {
    const auto &sp = rhs_arg_static_params_;
    auto is_narrow = [](data_type_t dt) {
        return utils::one_of(dt, data_type::s8, data_type::u8, data_type::f16,
                data_type::bf16);
    };
    // narrow_stage_vmm holds a narrow-dtype vector load; the scalar (FP-reg)
    // path bypasses it, except f16 which has no scalar path (always vector).
    // per_oc_spatial with channel lanes also broadcasts (scalar).
    const bool is_scalar_load
            = rhs_broadcasting_strategy == broadcasting_strategy_t::scalar
            || (rhs_broadcasting_strategy
                            == broadcasting_strategy_t::per_oc_spatial
                    && sp.per_oc_lanes_are_channels);
    const auto src1_dt = get_src1_desc(post_op, sp.dst_d).data_type;
    bool needs_narrow_stage = is_narrow(src1_dt)
            && !(is_scalar_load
                    && !utils::one_of(
                            src1_dt, data_type::f16, data_type::bf16));
    if (post_op.is_binary_with_ternary_op()) // src2 (no_broadcast) load
        needs_narrow_stage
                |= is_narrow(get_src2_desc(post_op, sp.dst_d).data_type);
    // gather_idx_vmm holds the per-lane index of a per_oc/per_oc_spatial gather
    // (unless the host resolves per-oc to contiguous channel lanes); per_mb_*
    // and per_w always gather.
    const bool needs_gather = utils::one_of(rhs_broadcasting_strategy,
                                      broadcasting_strategy_t::per_mb_spatial,
                                      broadcasting_strategy_t::per_mb_w,
                                      broadcasting_strategy_t::per_w)
            || (utils::one_of(rhs_broadcasting_strategy,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_oc_spatial)
                    && !sp.per_oc_lanes_are_channels);
    // narrow_stage_vmm doubles as the gather index-build scratch (v_low for a
    // blocked per_oc split, v_n for the per_mb/per_w flat-index math), so any
    // gather uses it regardless of dtype -- validate it there too.
    needs_narrow_stage |= needs_gather;
    if (!needs_narrow_stage && !needs_gather) return;

    // Each scratch is short-lived: it is consumed (staged then widened into the
    // helper, or built then consumed by the gather) before any v0-mask write,
    // so the v0 default is legal on its own. What must not happen is a scratch
    // overlapping a live accumulator or the helper; and a narrow gather reads
    // the index and writes the stage in one vluxei, so those two must be
    // disjoint (different-EEW indexed-load overlap is illegal). Like the helper,
    // an illegal host choice has no substitute and fails kernel creation. Every
    // group (helper, scratch, accumulator) is e32/acc_lmul, so group_stride.
    const int hint = sp.rhs_dt_helper_vmm_idx;
    const int n_vregs = jit_isa_traits_t<isa>::n_vregs;
    auto groups_overlap = [](int a, size_t sa, int b, size_t sb) {
        return a < b + (int)sb && b < a + (int)sa;
    };
    // Each scratch is a full e32/acc_lmul group (group_stride registers), so it
    // must be group-aligned and fit inside the register file. It also must not
    // be v0: v0 is the architectural mask register -- the select condition mask
    // lives there while a narrow/gather rhs load stages into this scratch, and
    // the min/max/compare paths write it -- so a scratch group containing v0
    // (base 0, which is also the unconfigured default in the static params)
    // clobbers it. Same rigor as validate_temp_vmm_hint's helper checks: an
    // illegal host choice has no substitute and fails kernel creation.
    auto scratch_ok = [&](int s) {
        if (s == 0) return false; // v0 = mask register / unconfigured default
        if (s % (int)group_stride != 0) return false; // group-aligned
        if (s + (int)group_stride > n_vregs) return false; // fits register file
        if (groups_overlap(s, group_stride, hint, group_stride)) return false;
        for (size_t base : vmm_idxs)
            if (groups_overlap(s, group_stride, (int)base, group_stride))
                return false;
        return true;
    };
    if (needs_narrow_stage)
        JIT_ASSERT(scratch_ok(sp.narrow_stage_vmm.getIdx()));
    if (needs_gather) JIT_ASSERT(scratch_ok(sp.gather_idx_vmm.getIdx()));
    if (needs_narrow_stage && needs_gather)
        JIT_ASSERT(!groups_overlap(sp.narrow_stage_vmm.getIdx(), group_stride,
                sp.gather_idx_vmm.getIdx(), group_stride));
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::validate_temp_vmm_hint(
        const injector_utils::vmm_index_set_t &vmm_idxs, size_t group_stride,
        int max_vmm_idx) const {
    const int hint = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    const int stride = (int)group_stride;
    // x64's adjust_temp_vmm_hint substitutes a free single vmm (0 or
    // max_vmm_idx) for an invalid hint and should_preserve_vmm spills it
    // locally around each use. Neither substitute is legal for the rv64 helper
    // group: v0 is the live mask register (the compare/select paths write it),
    // and the helper is an e32/acc_lmul group, so it must be group_stride-
    // aligned and fit in the register file. The helper is host-reserved by
    // contract, so an invalid hint is a host bug; JIT_ASSERT keeps the check in
    // Release builds, where the throw fails kernel creation via create_kernel()
    // instead of emitting code that corrupts the mask or an out-of-range group.
    const bool helper_group_legal = hint != 0 && hint % stride == 0
            && hint + stride - 1 <= max_vmm_idx;
    // The helper occupies [hint, hint + group_stride); it must not overlap any
    // processed accumulator's group [base, base + group_stride). Both intervals
    // are tested directly so a legal sparse set is not rejected on its endpoints
    // (e.g. m4 helper v8 with processed {v4, v12}: v4-v7 and v12-v15 straddle
    // v8-v11 but never touch it).
    bool overlaps_processed_vmms = false;
    for (const auto idx : vmm_idxs)
        overlaps_processed_vmms |= hint <= (int)idx + stride - 1
                && (int)idx <= hint + stride - 1;
    JIT_ASSERT(helper_group_legal && !overlaps_processed_vmms);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        size_t group_stride) const {
    // [start_idx, end_idx) is the half-open physical register span (x64
    // semantics); group_stride is the number of registers per accumulator (its
    // LMUL/EMUL). make_vmm_group_set steps by it, so a run of m>1 accumulators
    // yields its true group bases rather than x64's consecutive indices.
    compute_vector_range(injector_utils::make_vmm_group_set<isa>(
                                 start_idx, end_idx, group_stride),
            rhs_arg_idx, post_op, rhs_arg_params, group_stride);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        size_t group_stride) const {

    if (vmm_idxs.empty()) return;
    const auto start_idx = *(vmm_idxs.begin());

    // Phase 1 Validate the processed set and the temporary vmm user hint.
    // group_stride is each accumulator's e32 LMUL (1/2/4): the accumulators must
    // be group-aligned, non-overlapping, and in range, and the host-reserved
    // helper (also an e32/acc_lmul group) must not overlap them. Unlike x64,
    // which adjusts an invalid hint to a free single vmm, the rv64 helper has no
    // legal substitute, so a violation fails kernel creation.
    static constexpr int max_vmm_idx = jit_isa_traits_t<isa>::n_vregs - 1;
    const int vmm_hint = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    injector_utils::validate_vmm_group_set<isa>(vmm_idxs, group_stride);
    validate_temp_vmm_hint(vmm_idxs, group_stride, max_vmm_idx);

    const auto rhs_broadcasting_strategy
            = get_rhs_arg_broadcasting_strategy_or_hosted(
                    post_op.binary.src1_desc, rhs_arg_static_params_.dst_d,
                    supported_strategy_set_);
    const bool needs_ternary_input = post_op.is_binary_with_ternary_op();

    // Writes to x0 are discarded on RV64, so a helper GPR left at its default
    // would silently drop the address it is supposed to carry (no fault, just
    // a wrong rhs address). rhs_addr_reg always receives the rhs base;
    // rhs_helper_reg receives the computed offset for every non-scalar
    // strategy. This is checked here rather than at construction because the
    // supported-strategy set may legitimately be wider than the strategy this
    // entry actually resolves to (gnorm advertises {scalar} only and passes x0).
    JIT_ASSERT(rhs_arg_static_params_.rhs_addr_reg.getIdx() != x0.getIdx());
    JIT_ASSERT(IMPLICATION(
            rhs_broadcasting_strategy != broadcasting_strategy_t::scalar,
            rhs_arg_static_params_.rhs_helper_reg.getIdx() != x0.getIdx()));

    // Phase 2 Protect temporary registers content (x64's conditional guard
    // ladder; the scalar per-oc path additionally clobbers the calculate_*
    // scratch). The rhs dt helper vmm is spilled as a whole e32/acc_lmul group
    // (group_stride registers) when the host asks for it and this entry stages
    // through it (x64's preserve_vmm_helper && dt_helper_vmm_needed).
    const auto rhs_arg_data_type
            = get_src1_desc(post_op, rhs_arg_static_params_.dst_d).data_type;
    // The helper vmm is written unless the scalar strategy keeps the value in
    // the FP helper (.vf forms); f16 and the ternary condition always stage
    // through it, and so do min/max and the compare algs, whose ordered
    // compare-and-merge sequences materialize the scalar into a vector
    // operand (mirrors inject_binary/execute_binary's vrhs()).
    const bool alg_stages_through_helper_vmm = utils::one_of(post_op.binary.alg,
            alg_kind::binary_max, alg_kind::binary_min, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne);
    // The min/max/compare/select paths write the mask register v0 (vmflt/vmfeq
    // into VReg(0)) and then merge into dst; an accumulator group based at v0
    // (start_idx == 0, the only base whose group contains register 0) would
    // alias that mask. Non-mask algs (add/mul/sub/div) never touch v0, so v0 is
    // a legal accumulator for them. As with the helper hint, an illegal host
    // choice fails kernel creation rather than emitting mask-corrupting code.
    JIT_ASSERT(!((needs_ternary_input || alg_stages_through_helper_vmm)
            && start_idx == 0));
    // Phase 1b Validate the rv64-only rhs scratch groups this entry will use
    // (x64 has no analog -- its helper adjustment covers all temporaries).
    validate_rhs_scratch(
            vmm_idxs, group_stride, post_op, rhs_broadcasting_strategy);
    const bool dt_helper_vmm_needed = needs_ternary_input
            || alg_stages_through_helper_vmm
            || !(rhs_broadcasting_strategy == broadcasting_strategy_t::scalar
                    && !utils::one_of(rhs_arg_data_type, data_type::f16,
                            data_type::bf16));
    const bool use_oc_conversion_regs
            = utils::one_of(rhs_broadcasting_strategy,
                      broadcasting_strategy_t::per_oc,
                      broadcasting_strategy_t::per_oc_spatial)
            && rhs_arg_static_params_.per_oc_lanes_are_channels;
    const injector_utils::register_preserve_guard_t register_guard {host_,
            rhs_arg_static_params_.preserve_gpr_helpers
                    ? (use_oc_conversion_regs
                                      ? std::initializer_list<Xbyak_riscv::Reg>(
                                                {rhs_arg_static_params_
                                                                .rhs_addr_reg,
                                                        rhs_arg_static_params_
                                                                .rhs_helper_reg,
                                                        rhs_arg_static_params_
                                                                .rhs_addr_cache_reg,
                                                        X_TMP_0, X_TMP_1,
                                                        X_TMP_2, X_TMP_3,
                                                        X_TMP_4})
                                      : std::initializer_list<Xbyak_riscv::Reg>(
                                                {rhs_arg_static_params_
                                                                .rhs_addr_reg,
                                                        rhs_arg_static_params_
                                                                .rhs_helper_reg,
                                                        rhs_arg_static_params_
                                                                .rhs_addr_cache_reg,
                                                        X_TMP_0}))
                    : std::initializer_list<Xbyak_riscv::Reg>(),
            rhs_arg_static_params_.preserve_vmm_helper && dt_helper_vmm_needed
                    ? std::initializer_list<Xbyak_riscv::VReg>(
                              {Xbyak_riscv::VReg(vmm_hint)})
                    : std::initializer_list<Xbyak_riscv::VReg>(),
            {} /*freg*/, group_stride};

    rhs_address_t rhs_arg_addr(Xbyak_riscv::Reg(0));

    // Phase 3 Apply binary post-op over all vmms. rhs_arg_params_differ compares
    // this accumulator's params against the previously processed one to decide
    // whether the rhs address must be rebuilt. Unlike x64 -- where accumulators
    // are consecutive vmm indices, so vmm_idx - 1 is the previous one -- an rv64
    // run of m>1 accumulators steps its group bases by group_stride, so the
    // previous base is not vmm_idx - 1; track it explicitly (the ordered set
    // iterates ascending).
    size_t prev_vmm_idx = start_idx;
    for (const auto vmm_idx : vmm_idxs) {
        // For binary ops with ternary inputs the select condition src2 is
        // loaded first and immediately consumed into the v0 mask: rv64 cannot
        // stack-spill a vector like x64's push_vmm, so the mask register is
        // the only carrier that survives the src1 load reusing the same
        // helper vmm. The condition supports no broadcast (x64 contract), so
        // its address follows the no_broadcast offset path.
        if (needs_ternary_input) {
            const auto rhs2_arg_data_type
                    = get_src2_desc(post_op, rhs_arg_static_params_.dst_d)
                              .data_type;
            const auto rhs2_arg_addr = prepare_rhs_arg_addr(vmm_idx,
                    rhs_arg_idx + 1, post_op, rhs_arg_params,
                    broadcasting_strategy_t::no_broadcast, true);

            const Vmm tern_tmp_vmm
                    = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);
            load_rhs(rhs2_arg_data_type, tern_tmp_vmm, rhs2_arg_addr,
                    group_stride);
            if (rhs2_arg_data_type != data_type::f32
                    && !utils::one_of(rhs2_arg_data_type, data_type::f16,
                            data_type::bf16))
                cvt_to_f32(tern_tmp_vmm);
            // v0 = (cond == 0): the lanes where the select picks src1. NaN
            // compares unequal to 0, so a NaN condition keeps the
            // accumulator, matching bool(NaN) == true in the reference.
            const auto &tmp_freg = rhs_arg_static_params_.rhs_dt_helper_freg;
            host_->fmv_w_x(tmp_freg, x0);
            host_->vmfeq_vf(VReg(0), tern_tmp_vmm, tmp_freg);
        }

        // The src1 address must also be rebuilt whenever the ternary
        // condition was prepared: both operands share rhs_addr_reg (x64
        // reloads only under the scalar strategy because its other
        // strategies keep cached addresses).
        if (vmm_idx == start_idx || needs_ternary_input
                || rhs_arg_params_differ(vmm_idx, prev_vmm_idx, rhs_arg_params,
                        rhs_broadcasting_strategy)) {
            rhs_arg_addr = prepare_rhs_arg_addr(vmm_idx, rhs_arg_idx, post_op,
                    rhs_arg_params, rhs_broadcasting_strategy, false);
        }

        if (needs_ternary_input)
            inject_binary_with_ternary_op(
                    post_op, Vmm(vmm_idx), rhs_arg_addr, group_stride);
        else
            inject_binary(post_op, Vmm(vmm_idx), rhs_arg_addr, group_stride);

        prev_vmm_idx = vmm_idx;
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector(size_t idx,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        size_t group_stride) const {
    compute_vector_range(
            {idx}, rhs_arg_idx, post_op, rhs_arg_params, group_stride);
}

template <cpu_isa_t isa>
rhs_address_t jit_uni_binary_injector_t<isa>::prepare_rhs_arg_addr(
        std::size_t vmm_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        const broadcasting_strategy_t rhs_broadcasting_strategy,
        bool is_ternary_input) const {

    static constexpr auto rhs_arg_ptr_size = sizeof(const void *);
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const auto &rhs_helper_reg = rhs_arg_static_params_.rhs_helper_reg;
    const auto &rhs_md = is_ternary_input
            ? get_src2_desc(post_op, rhs_arg_static_params_.dst_d)
            : get_src1_desc(post_op, rhs_arg_static_params_.dst_d);
    const auto rhs_arg_elem_size = types::data_type_size(rhs_md.data_type);

    // x64: param1 is the kernel abi param (the args struct), abi_param_offset
    // the rhs pointer-array field, dereferenced twice. x64 gates the reload on
    // is_first (its addresses are cached); rv64 caches no addresses and
    // reloads on every call.
    host_->ld(rhs_addr_reg, param1_,
            (int)rhs_arg_static_params_.abi_param_offset);
    host_->ld(
            rhs_addr_reg, rhs_addr_reg, (int)(rhs_arg_idx * rhs_arg_ptr_size));

    switch (rhs_broadcasting_strategy) {
        case broadcasting_strategy_t::scalar:
            return rhs_address_t(rhs_addr_reg, 0, true);
        case broadcasting_strategy_t::no_broadcast: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_out_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_out_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_out_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_no_broadcast_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            if (rhs_arg_static_params_.is_strided)
                return rhs_address_t(rhs_addr_reg, 0, false, false,
                        true /*isStrided*/, rhs_arg_static_params_.rhs_stride);
            return rhs_address_t(rhs_addr_reg);
        }
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            const bool is_per_oc_spatial = rhs_broadcasting_strategy
                    == broadcasting_strategy_t::per_oc_spatial;
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_oc_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_oc_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_oc_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_oc_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size,
                    is_per_oc_spatial);

            // Channel-aligned host: aarch64's per-call addressing (broadcast
            // for per_oc_spatial, contiguous otherwise); else both are
            // realized by the per-lane gather (see append_oc_offset).
            if (rhs_arg_static_params_.per_oc_lanes_are_channels)
                return rhs_address_t(rhs_addr_reg, 0, is_per_oc_spatial);
            return rhs_address_t(rhs_addr_reg, 0, false, true /*isGather*/);
        }
        case broadcasting_strategy_t::per_mb_spatial: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_sp_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_sp_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_sp_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_mb_sp_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            return rhs_address_t(rhs_addr_reg, 0, false, true /*isGather*/);
        }
        case broadcasting_strategy_t::per_mb_w: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_mb_w_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_mb_w_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_mb_w_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_mb_w_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            return rhs_address_t(rhs_addr_reg, 0, false, true /*isGather*/);
        }
        case broadcasting_strategy_t::per_w: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_w_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_w_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_w_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_w_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            return rhs_address_t(rhs_addr_reg, 0, false, true /*isGather*/);
        }
        default: assert(false && "Broadcasting type not supported");
    }

    return rhs_address_t(rhs_addr_reg, 0, true);
}

// --- offset helpers (mirror aarch64 append_offset_from_operand /
// append_offset_under_mem_addr / append_value_offset). rv64 consumers use the
// out_reg + out_elem_off_val path, so these three are structural parity and
// no-ops in the current consumers (is_dst_orig_set gates them out, matching
// aarch64). ---

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_from_operand(
        const std::map<int, rhs_operand_t> &vmm_idx_to_elem_operand_off,
        int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
        const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes) const {
    const auto it_operand_off = vmm_idx_to_elem_operand_off.find(vmm_idx);
    if (it_operand_off != vmm_idx_to_elem_operand_off.end()
            && !rhs_arg_static_params_.is_dst_orig_set()) {
        const auto &op = it_operand_off->second;
        if (op.isAddress_)
            host_->ld(tmp_reg, op.address_.base_, (int)op.address_.offt_);
        else
            host_->mv(tmp_reg, Xbyak_riscv::Reg(op.idx_));
        if (elem_size_bytes > 1)
            host_->slli(tmp_reg, tmp_reg, (int)std::log2(elem_size_bytes));
        host_->add(addr_reg, addr_reg, tmp_reg);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_under_mem_addr(
        const std::map<int, rhs_address_t> &vmm_idx_to_elem_addr_off,
        int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
        const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes) const {
    const auto it_off_addr = vmm_idx_to_elem_addr_off.find(vmm_idx);
    if (it_off_addr != vmm_idx_to_elem_addr_off.end()
            && !rhs_arg_static_params_.is_dst_orig_set()) {
        host_->ld(tmp_reg, it_off_addr->second.base_,
                (int)it_off_addr->second.offt_);
        if (elem_size_bytes > 1)
            host_->slli(tmp_reg, tmp_reg, (int)std::log2(elem_size_bytes));
        host_->add(addr_reg, addr_reg, tmp_reg);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_value_offset(
        const std::map<int, size_t> &vmm_idx_to_elem_val_off, int vmm_idx,
        const Xbyak_riscv::Reg &addr_reg, std::size_t elem_size_bytes) const {
    const auto it_off_val = vmm_idx_to_elem_val_off.find(vmm_idx);
    if (it_off_val != vmm_idx_to_elem_val_off.end()
            && !rhs_arg_static_params_.is_dst_orig_set()) {
        const auto &t = X_TMP_0;
        host_->load_imm64(t, (int64_t)(it_off_val->second * elem_size_bytes));
        host_->add(addr_reg, addr_reg, t);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_no_broadcast_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_riscv::Reg &addr_reg, const Xbyak_riscv::Reg &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        rhs_address_t out_addr = is_out_addr
                ? it_out_addr->second
                : rhs_address_t(it_out_reg->second);
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        calculate_no_broadcast(out_addr,
                it_off_val != vmm_idx_to_out_elem_off_val.end()
                        ? it_off_val->second
                        : 0,
                tmp_reg);

        // byte-offset host (pooling): the byte offset is added straight to the
        // rhs base (elements already scaled by the caller).
        if (rhs_arg_static_params_.off_is_bytes) {
            host_->add(addr_reg, addr_reg, tmp_reg);
        } else {
            if (elem_size_bytes > 1)
                host_->slli(tmp_reg, tmp_reg, (int)std::log2(elem_size_bytes));
            host_->add(addr_reg, addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_no_broadcast(rhs_address_t addr,
        std::size_t offset, const Xbyak_riscv::Reg &out_reg) const {
    const auto &t0 = X_TMP_0;
    if (!rhs_arg_static_params_.is_dst_orig_set()) {
        // Host-positioned mode (rv64-only, null dst_d): addr.getBase()
        // already holds the running element offset the VLA host maintains (a
        // byte offset under off_is_bytes); no address recovery is needed.
        if (offset > 0) {
            host_->load_imm64(t0, (int64_t)offset);
            host_->add(out_reg, addr.getBase(), t0);
        } else
            host_->mv(out_reg, addr.getBase());
        return;
    }
    // aarch64: recover the dst element offset from the dst address,
    // (addr + offt + offset - dst_orig) >> log2(dst data type size), with
    // dst_orig loaded from param1 + dst_orig_offset.
    if (addr.offt_ != 0) {
        host_->load_imm64(t0, addr.offt_);
        host_->add(out_reg, addr.getBase(), t0);
    } else
        host_->mv(out_reg, addr.getBase());
    if (offset > 0) {
        host_->load_imm64(t0, (int64_t)offset);
        host_->add(out_reg, out_reg, t0);
    }
    host_->ld(t0, param1_, (int)rhs_arg_static_params_.dst_orig_offset);
    host_->sub(out_reg, out_reg, t0);
    host_->srli(out_reg, out_reg,
            (uint32_t)std::log2(types::data_type_size(
                    rhs_arg_static_params_.dst_d.data_type())));
}

// Build the per-lane byte-index gather vector (gather_idx_vmm) that realizes a
// per-channel rhs load under VLA: a vl-run may span the channel dimension, so
// each lane resolves its own index. The compute vtype (e32/m4) must be active.
// `flat_off_reg` (tmp_reg) holds the run's first-lane element offset.
//   idx_bytes[lane] = (((o+lane) / oc_stride) % oc_count) * blk + ((o+lane) % blk)
//                     scaled by elem_size.
template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_oc_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_riscv::Reg &addr_reg, const Xbyak_riscv::Reg &tmp_reg,
        std::size_t elem_size_bytes, bool is_per_oc_spatial) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);
    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();
    if (!(is_out_addr || is_out_reg)) return;

    rhs_address_t out_addr = is_out_addr ? it_out_addr->second
                                         : rhs_address_t(it_out_reg->second);
    const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
    calculate_no_broadcast(out_addr,
            it_off_val != vmm_idx_to_out_elem_off_val.end() ? it_off_val->second
                                                            : 0,
            tmp_reg); // tmp_reg = flat element offset o

    const auto dst_d = rhs_arg_static_params_.dst_d;

    if (rhs_arg_static_params_.per_oc_lanes_are_channels) {
        // aarch64's scalar path: derive the channel (block) offset from the
        // recovered element offset and add it to the rhs base. The host
        // guarantees every vector's lanes stay within the addressed channel
        // row/block (see static params), so the rhs is then a contiguous
        // slice (per_oc) or a single value (per_oc_spatial).
        MAYBE_UNUSED(is_per_oc_spatial);
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);
        // get_layout_type() classifies a blocked dst as c_blocked only when the
        // single inner block is the channel one; anything else (e.g. Abcd16a,
        // blocked on N) is `unsupported` and has no valid channel formula here.
        // The host's post_ops_ok must have rejected it -- fail kernel creation
        // rather than emit garbage addresses in a Release build.
        JIT_ASSERT(layout != injector_utils::layout_t::unsupported);
        switch (layout) {
            case injector_utils::layout_t::ncsp:
                calculate_oc_ncsp(strides, tmp_reg);
                break;
            case injector_utils::layout_t::c_blocked:
                calculate_oc_blocked(strides, tmp_reg);
                break;
            case injector_utils::layout_t::nspc:
                calculate_oc_nspc(strides, tmp_reg);
                break;
            case injector_utils::layout_t::cspn:
                calculate_oc_cspn(strides, tmp_reg);
                break;
            default: assert(!"Unknown layout");
        }

        if (elem_size_bytes == 1) {
            host_->add(addr_reg, addr_reg, X_TMP_0);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mv(tmp_reg, X_TMP_0);
            host_->slli(tmp_reg, tmp_reg, shift_val);
            host_->add(addr_reg, addr_reg, tmp_reg);
        }
        return;
    }

    // Otherwise both per_oc and per_oc_spatial are realized by the per-lane
    // gather: under RVV VLA an unrouted run spans channel boundaries even for
    // per_oc_spatial layouts, so each lane resolves its own channel.
    MAYBE_UNUSED(is_per_oc_spatial);

    // per_oc gather. Derive (oc_count, oc_stride, blk) for the channel dim.
    const auto &bd = dst_d.blocking_desc();
    dim_t blk = 1;
    for (int k = 0; k < bd.inner_nblks; k++)
        if (bd.inner_idxs[k] == 1) blk *= bd.inner_blks[k];
    const dim_t oc_count = (dst_d.dims()[1] + blk - 1) / blk;
    const dim_t oc_stride = bd.strides[1];

    const auto &v_idx = rhs_arg_static_params_.gather_idx_vmm;
    const auto &v_low = rhs_arg_static_params_.narrow_stage_vmm;
    const auto &t = X_TMP_0;
    host_->vid_v(v_idx);
    host_->vadd_vx(v_idx, v_idx, tmp_reg); // o + lane
    if (blk > 1) host_->vand_vi(v_low, v_idx, (int)(blk - 1)); // low = o % blk
    if (oc_stride != 1) {
        host_->load_imm64(t, oc_stride);
        host_->vdivu_vx(v_idx, v_idx, t);
    }
    host_->load_imm64(t, oc_count);
    host_->vremu_vx(v_idx, v_idx, t); // outer index
    if (blk > 1) {
        int lg = 0;
        for (dim_t b = blk; b > 1; b >>= 1)
            lg++;
        host_->vsll_vi(v_idx, v_idx, lg); // outer * blk
        host_->vadd_vv(v_idx, v_idx, v_low); // + low
        // A padded channel tail makes the reconstructed index run up to
        // padded_C - 1 while the per_oc rhs only holds the logical C elements,
        // so clamp before scaling to bytes. This bounds the gather; the lanes
        // that get clamped are padding, and the HOST still owns making their
        // result harmless (mask them off, or zero the padded dst tail as the
        // binary tail kernel does).
        host_->load_imm64(t, dst_d.dims()[1] - 1);
        host_->vminu_vx(v_idx, v_idx, t);
    }
    if (elem_size_bytes > 1)
        host_->vsll_vi(v_idx, v_idx, (int)std::log2(elem_size_bytes));
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_oc_ncsp(const dim_t *strides,
        const Xbyak_riscv::Reg &tmp_reg, const bool residue) const {
    // c = (offset % strides[0]) / strides[1]; output = x_tmp0
    const auto &x0t = X_TMP_0;
    const auto &x1t = X_TMP_1;
    const auto &x2t = X_TMP_2;
    const auto &x3t = X_TMP_3;
    const auto &x4t = X_TMP_4;

    host_->load_imm64(x3t, strides[0]);
    host_->load_imm64(x4t, strides[1]);
    host_->remu(x2t, tmp_reg, x3t);
    if (residue) {
        host_->divu(x0t, x2t, x4t);
        host_->remu(x1t, x2t, x4t);
    } else
        host_->divu(x0t, x2t, x4t);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_oc_blocked(
        const dim_t *strides, const Xbyak_riscv::Reg &tmp_reg) const {
    // c = ((offset % strides[0]) / strides[1]) * blk + offset % blk_size
    const auto dst_d = rhs_arg_static_params_.dst_d;
    // Unlike the gather path -- which folds every channel-dim inner block
    // (`for k: if inner_idxs[k] == 1 blk *= inner_blks[k]`) -- this formula
    // assumes a single channel block. Keep the two in agreement by requiring
    // that shape here; the hosts' post_ops_ok already gate it (see x64's
    // identical blocked-format check), and a violation would silently compute a
    // wrong channel rather than fault.
    JIT_ASSERT(dst_d.blocking_desc().inner_nblks == 1
            && dst_d.blocking_desc().inner_idxs[0] == 1);
    const int blk_size = dst_d.blocking_desc().inner_blks[0];
    const auto &x0t = X_TMP_0;
    const auto &x1t = X_TMP_1;
    const auto &x2t = X_TMP_2;
    const auto &x3t = X_TMP_3;

    calculate_oc_ncsp(strides, tmp_reg, /*residue=*/true);
    // extract c % blk_size
    host_->load_imm64(x3t, blk_size);
    host_->remu(x2t, x1t, x3t);

    host_->load_imm64(tmp_reg, blk_size);
    host_->mul(x0t, x0t, tmp_reg);
    host_->add(x0t, x0t, x2t);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_oc_nspc(
        const dim_t *strides, const Xbyak_riscv::Reg &tmp_reg) const {
    // c = offset % C; output = x_tmp0
    const auto &x0t = X_TMP_0;
    const auto &x1t = X_TMP_1;
    const auto C = rhs_arg_static_params_.dst_d.dims()[1];
    host_->load_imm64(x1t, C);
    host_->remu(x0t, tmp_reg, x1t);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_oc_cspn(
        const dim_t *strides, const Xbyak_riscv::Reg &tmp_reg) const {
    // c = offset / strides[1]; output = x_tmp0
    const auto &x0t = X_TMP_0;
    const auto &x1t = X_TMP_1;
    host_->load_imm64(x1t, strides[1]);
    host_->divu(x0t, tmp_reg, x1t);
}

// --- per_mb_spatial / per_mb_w: rhs broadcasts the channel (rhs [N,1,D,H,W] or
// [N,1,1,1,W], dense/plain). Under VLA a run spans the broadcast dim, so both
// are realized by the per-lane gather: each lane's plain rhs element index is
// computed from its flat dst offset with vector div/mod. This is the same index
// the scalar calculate_mb_* methods compute per vmm, generalized to per-lane
// and to all four dst layouts. per_mb_spatial: rhs_idx = n*sp_size + spatial
// (sp_size = product of spatial dims); per_mb_w: rhs_idx = n*W + w. Uses
// gather_idx_vmm + narrow_stage_vmm as the two vector temporaries (the
// narrow-load staging is temporally free during index build) and x_tmp0 as the
// scalar constant register. ---

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_mb_sp_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_riscv::Reg &addr_reg, const Xbyak_riscv::Reg &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);
    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();
    if (!(is_out_addr || is_out_reg)) return;

    rhs_address_t out_addr = is_out_addr ? it_out_addr->second
                                         : rhs_address_t(it_out_reg->second);
    const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
    calculate_no_broadcast(out_addr,
            it_off_val != vmm_idx_to_out_elem_off_val.end() ? it_off_val->second
                                                            : 0,
            tmp_reg); // tmp_reg = flat element offset o

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto strides = dst_d.blocking_desc().strides;
    const auto layout = injector_utils::get_layout_type(dst_d);

    const auto &v_idx = rhs_arg_static_params_.gather_idx_vmm;
    const auto &v_n = rhs_arg_static_params_.narrow_stage_vmm;
    const auto &t = X_TMP_0;
    host_->vid_v(v_idx);
    host_->vadd_vx(v_idx, v_idx, tmp_reg); // o + lane
    // Gated to plain ncsp/nspc by is_bcast_supported (a blocked/cspn dst
    // resolves the `any` rhs to a matching non-plain layout the plain index
    // below does not model).
    if (layout == injector_utils::layout_t::ncsp) {
        // rhs_idx = n*sp_size + spatial, sp_size == strides[1] (channel stride).
        host_->load_imm64(t, strides[0]);
        host_->vdivu_vx(v_n, v_idx, t); // n
        host_->vremu_vx(v_idx, v_idx, t); // within = o % strides[0]
        host_->load_imm64(t, strides[1]);
        host_->vremu_vx(v_idx, v_idx, t); // spatial = within % strides[1]
        host_->vmul_vx(v_n, v_n, t); // n * sp_size
        host_->vadd_vv(v_idx, v_idx, v_n);
    } else { // nspc: channel innermost -> rhs_idx = o / C_padded
        host_->load_imm64(t, dst_d.padded_dims()[1]);
        host_->vdivu_vx(v_idx, v_idx, t);
    }
    if (elem_size_bytes > 1)
        host_->vsll_vi(v_idx, v_idx, (int)std::log2(elem_size_bytes));
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_mb_w_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_riscv::Reg &addr_reg, const Xbyak_riscv::Reg &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);
    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();
    if (!(is_out_addr || is_out_reg)) return;

    rhs_address_t out_addr = is_out_addr ? it_out_addr->second
                                         : rhs_address_t(it_out_reg->second);
    const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
    calculate_no_broadcast(out_addr,
            it_off_val != vmm_idx_to_out_elem_off_val.end() ? it_off_val->second
                                                            : 0,
            tmp_reg); // tmp_reg = flat element offset o

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto strides = dst_d.blocking_desc().strides;
    const auto ndims = dst_d.ndims();
    const dim_t W = dst_d.dims()[ndims - 1];
    const auto layout = injector_utils::get_layout_type(dst_d);

    const auto &v_idx = rhs_arg_static_params_.gather_idx_vmm;
    const auto &v_n = rhs_arg_static_params_.narrow_stage_vmm;
    const auto &t = X_TMP_0;
    host_->vid_v(v_idx);
    host_->vadd_vx(v_idx, v_idx, tmp_reg); // o + lane
    // rhs [N,1,1,1,W]: rhs_idx = n * W + w. Gated to plain ncsp/nspc.
    host_->load_imm64(t, strides[0]);
    host_->vdivu_vx(v_n, v_idx, t); // n = o / strides[0]
    if (layout == injector_utils::layout_t::ncsp) {
        // w = o % strides[ndims-2] (the last dim's parent stride == W; the
        // innermost dim has stride 1).
        host_->load_imm64(t, strides[ndims - 2]);
        host_->vremu_vx(v_idx, v_idx, t); // w
    } else { // nspc: channel innermost -> w = (o / C_padded) % W
        host_->load_imm64(t, dst_d.padded_dims()[1]);
        host_->vdivu_vx(v_idx, v_idx, t); // spatial-flat
        host_->load_imm64(t, W);
        host_->vremu_vx(v_idx, v_idx, t); // w
    }
    host_->load_imm64(t, W);
    host_->vmul_vx(v_n, v_n, t); // n * W
    host_->vadd_vv(v_idx, v_idx, v_n);
    if (elem_size_bytes > 1)
        host_->vsll_vi(v_idx, v_idx, (int)std::log2(elem_size_bytes));
}

// --- per_w (advertised; realized via the RVV gather) ---

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_w_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_riscv::Reg &addr_reg, const Xbyak_riscv::Reg &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);
    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();
    if (!(is_out_addr || is_out_reg)) return;

    rhs_address_t out_addr = is_out_addr ? it_out_addr->second
                                         : rhs_address_t(it_out_reg->second);
    const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
    calculate_no_broadcast(out_addr,
            it_off_val != vmm_idx_to_out_elem_off_val.end() ? it_off_val->second
                                                            : 0,
            tmp_reg); // tmp_reg = flat element offset o

    // Per-lane gather over the W dim: idx[lane] = ((o + lane) / w_stride) % W.
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto strides = dst_d.blocking_desc().strides;
    const int wd = dst_d.ndims() - 1;
    const dim_t W = dst_d.dims()[wd];
    const dim_t w_stride = strides[wd];

    const auto &v_idx = rhs_arg_static_params_.gather_idx_vmm;
    const auto &t = X_TMP_0;
    host_->vid_v(v_idx);
    host_->vadd_vx(v_idx, v_idx, tmp_reg); // o + lane
    if (w_stride != 1) {
        host_->load_imm64(t, w_stride);
        host_->vdivu_vx(v_idx, v_idx, t);
    }
    host_->load_imm64(t, W);
    host_->vremu_vx(v_idx, v_idx, t);
    if (elem_size_bytes > 1)
        host_->vsll_vi(v_idx, v_idx, (int)std::log2(elem_size_bytes));
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::inject_binary(
        const dnnl_post_ops::entry_t &post_op, Vmm dst,
        const rhs_address_t &rhs_addr, size_t group_stride) const {

    const auto &alg = post_op.binary.alg;
    const auto &rhs_arg_data_type = post_op.binary.src1_desc.data_type;

    // The rv64 counterpart of x64's process_rhs_arg_using_tmp_vmm decision: a
    // broadcast scalar stays in the FP helper and the .vf instruction forms
    // consume it directly (f16 excepted -- its conversion needs the vector
    // unit); everything else (per_element/gather) loads into the vmm helper.
    if (rhs_addr.isBroadcast()
            && !utils::one_of(
                    rhs_arg_data_type, data_type::f16, data_type::bf16)) {
        const auto &tmp_freg = rhs_arg_static_params_.rhs_dt_helper_freg;
        execute_broadcast(rhs_arg_data_type, tmp_freg, rhs_addr);
        execute_binary(alg, dst, dst, tmp_freg);
    } else {
        const Vmm tmp_vmm = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);
        load_rhs(rhs_arg_data_type, tmp_vmm, rhs_addr, group_stride);
        if (rhs_arg_data_type != data_type::f32
                && !utils::one_of(
                        rhs_arg_data_type, data_type::f16, data_type::bf16))
            cvt_to_f32(tmp_vmm);
        execute_binary(alg, dst, dst, tmp_vmm);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::inject_binary_with_ternary_op(
        const dnnl_post_ops::entry_t &post_op, Vmm dst,
        const rhs_address_t &rhs_addr, size_t group_stride) const {
    const auto alg = post_op.binary.alg;
    const auto rhs_arg_data_type
            = get_src1_desc(post_op, rhs_arg_static_params_.dst_d).data_type;

    if (alg == alg_kind::binary_select) {
        // dst = cond ? dst : src1 with v0 = (cond == 0) prepared by the
        // caller. RVV selects with a single mask merge, so unlike x64's
        // AND/blend spill sequence no extra register state is needed: a
        // scalar src1 merges via vfmerge.vfm straight from the FP helper, a
        // vector src1 via vmerge.vvm from the loaded helper vmm.
        if (rhs_addr.isBroadcast()
                && !utils::one_of(
                        rhs_arg_data_type, data_type::f16, data_type::bf16)) {
            const auto &tmp_freg = rhs_arg_static_params_.rhs_dt_helper_freg;
            execute_broadcast(rhs_arg_data_type, tmp_freg, rhs_addr);
            host_->vfmerge_vfm(dst, dst, tmp_freg);
        } else {
            const Vmm tmp_vmm
                    = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);
            load_rhs(rhs_arg_data_type, tmp_vmm, rhs_addr, group_stride);
            if (rhs_arg_data_type != data_type::f32
                    && !utils::one_of(
                            rhs_arg_data_type, data_type::f16, data_type::bf16))
                cvt_to_f32(tmp_vmm);
            host_->vmerge_vvm(dst, dst, tmp_vmm);
        }
    } else {
        assert(!"unsupported algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_cmp_mask(
        const Vmm &lhs, const Vmm &rhs, const alg_kind_t cmp_alg) const {
    using namespace alg_kind;
    switch (cmp_alg) {
        case binary_ge: host_->vmfge_vv(VReg(0), lhs, rhs); break;
        case binary_gt: host_->vmfgt_vv(VReg(0), lhs, rhs); break;
        case binary_le: host_->vmfle_vv(VReg(0), lhs, rhs); break;
        case binary_lt: host_->vmflt_vv(VReg(0), lhs, rhs); break;
        case binary_eq: host_->vmfeq_vv(VReg(0), lhs, rhs); break;
        case binary_ne: host_->vmfne_vv(VReg(0), lhs, rhs); break;
        default: assert(!"unsupported comparison algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_cmp_binary(const Vmm &dst,
        const Vmm &lhs, const Vmm &rhs, const alg_kind_t cmp_alg) const {
    compute_cmp_mask(lhs, rhs, cmp_alg);
    // Materialize the oneDNN-required f32 0/1 result under the v0 mask (x64
    // blends 1.0f from a preloaded register instead). The 1.0f constant goes
    // through the x_tmp0 scratch: rhs_addr_reg must stay intact so a shared
    // rhs address remains valid across the vmm loop.
    const auto &tmp_freg = rhs_arg_static_params_.rhs_dt_helper_freg;
    const auto &tmp_reg = X_TMP_0;
    host_->fmv_w_x(tmp_freg, x0);
    host_->vfmv_v_f(dst, tmp_freg);
    host_->li(tmp_reg, 0x3f800000);
    host_->fmv_w_x(tmp_freg, tmp_reg);
    host_->vfmerge_vfm(dst, dst, tmp_freg);
}

// .vf (scalar FP rhs) / .vv (vector rhs) dispatch. x64 funnels both through one
// `uni_v*ps(dst, lhs, rhs)` because every x86 instruction takes a unified
// Operand; on rv64 the two forms are distinct mnemonics and FReg/VReg share no
// base class, so overload resolution does the dispatch. (The project builds as
// C++11, so `if constexpr` is not available and a reinterpret_cast between the
// two register types would be a strict-aliasing violation.)
template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::emit_arith(alg_kind_t alg, const Vmm &dst,
        const Vmm &lhs, const Xbyak_riscv::FReg &rhs) const {
    switch (alg) {
        case alg_kind::binary_add: host_->vfadd_vf(dst, lhs, rhs); break;
        case alg_kind::binary_sub: host_->vfsub_vf(dst, lhs, rhs); break;
        case alg_kind::binary_mul: host_->vfmul_vf(dst, lhs, rhs); break;
        case alg_kind::binary_div: host_->vfdiv_vf(dst, lhs, rhs); break;
        default: assert(!"unsupported algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::emit_arith(
        alg_kind_t alg, const Vmm &dst, const Vmm &lhs, const Vmm &rhs) const {
    switch (alg) {
        case alg_kind::binary_add: host_->vfadd_vv(dst, lhs, rhs); break;
        case alg_kind::binary_sub: host_->vfsub_vv(dst, lhs, rhs); break;
        case alg_kind::binary_mul: host_->vfmul_vv(dst, lhs, rhs); break;
        case alg_kind::binary_div: host_->vfdiv_vv(dst, lhs, rhs); break;
        default: assert(!"unsupported algorithm");
    }
}

template <cpu_isa_t isa>
typename jit_uni_binary_injector_t<isa>::Vmm
jit_uni_binary_injector_t<isa>::materialize_rhs(
        const Xbyak_riscv::FReg &rhs) const {
    const Vmm helper_vmm = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);
    host_->vfmv_v_f(helper_vmm, rhs);
    return helper_vmm;
}

template <cpu_isa_t isa>
typename jit_uni_binary_injector_t<isa>::Vmm
jit_uni_binary_injector_t<isa>::materialize_rhs(const Vmm &rhs) const {
    return rhs;
}

template <cpu_isa_t isa>
template <typename T>
void jit_uni_binary_injector_t<isa>::execute_binary(alg_kind_t binary_alg,
        const Vmm &dst, const Vmm &lhs, const T &rhs) const {
    switch (binary_alg) {
        case alg_kind::binary_add:
        case alg_kind::binary_sub:
        case alg_kind::binary_mul:
        case alg_kind::binary_div:
            // The plain arithmetic ops keep the .vf form for a scalar rhs.
            emit_arith(binary_alg, dst, lhs, rhs);
            break;
        case alg_kind::binary_max: {
            // nstl::max(lhs,rhs) = (rhs < lhs) ? lhs : rhs (picks rhs on
            // ties/unordered, matching the reference and x86 vmaxps).
            const Vmm r = materialize_rhs(rhs);
            host_->vmflt_vv(VReg(0), r, lhs);
            host_->vmerge_vvm(dst, r, lhs);
            break;
        }
        case alg_kind::binary_min: {
            // nstl::min(lhs,rhs) = (lhs < rhs) ? lhs : rhs.
            const Vmm r = materialize_rhs(rhs);
            host_->vmflt_vv(VReg(0), lhs, r);
            host_->vmerge_vvm(dst, r, lhs);
            break;
        }
        // The compare sequences need a vector operand as well.
        case alg_kind::binary_ge:
        case alg_kind::binary_gt:
        case alg_kind::binary_le:
        case alg_kind::binary_lt:
        case alg_kind::binary_eq:
        case alg_kind::binary_ne:
            execute_cmp_binary(dst, lhs, materialize_rhs(rhs), binary_alg);
            break;
        default: assert(!"unsupported algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast(
        const data_type_t &data_type, const Xbyak_riscv::FReg &tmp_freg,
        const rhs_address_t &rhs_addr) const {
    assert(is_data_supported(isa, data_type) && "unsupported data type");
    const auto &base = rhs_addr.getBase();
    const int offt = (int)rhs_addr.offt_;
    // The integer value stages through x_tmp0 (aarch64 uses its reserved
    // W_TMP_0): the base register must stay intact so a shared scalar address
    // remains valid across the vmm loop (the scalar strategy never recomputes
    // it).
    const auto &tmp_reg = X_TMP_0;
    switch (data_type) {
        case data_type::f32: host_->flw(tmp_freg, base, offt); break;
        case data_type::s32:
            host_->lw(tmp_reg, base, offt);
            host_->fcvt_s_w(tmp_freg, tmp_reg);
            break;
        case data_type::s8:
            host_->lb(tmp_reg, base, offt);
            host_->fcvt_s_w(tmp_freg, tmp_reg);
            break;
        case data_type::u8:
            host_->lbu(tmp_reg, base, offt);
            host_->fcvt_s_wu(tmp_freg, tmp_reg);
            break;
        default: assert(!"unsupported data type");
    }
}

// The accumulator computes at e32/acc_lmul, where acc_lmul is encoded by
// group_stride (the accumulator's register count: 1->m1, 2->m2, 4->m4). A
// narrower rhs must be loaded at the LMUL that shares the accumulator's VLMAX so
// that `vsetvli x0,x0` preserves vl and the widen lands in e32/acc_lmul: e16
// (f16) at acc_lmul/2 and e8 (s8/u8) at acc_lmul/4, taking a fractional LMUL
// when the accumulator is below m4. m8 is unsupported (see make_vmm_group_set).
static Xbyak_riscv::LMUL acc_e32_lmul(size_t group_stride) {
    switch (group_stride) {
        case 1: return Xbyak_riscv::LMUL::m1;
        case 2: return Xbyak_riscv::LMUL::m2;
        default: return Xbyak_riscv::LMUL::m4;
    }
}
static Xbyak_riscv::LMUL rhs_e16_lmul(size_t group_stride) {
    switch (group_stride) {
        case 1: return Xbyak_riscv::LMUL::mf2;
        case 2: return Xbyak_riscv::LMUL::m1;
        default: return Xbyak_riscv::LMUL::m2;
    }
}
static Xbyak_riscv::LMUL rhs_e8_lmul(size_t group_stride) {
    switch (group_stride) {
        case 1: return Xbyak_riscv::LMUL::mf4;
        case 2: return Xbyak_riscv::LMUL::mf2;
        default: return Xbyak_riscv::LMUL::m1;
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs(const data_type_t &data_type,
        const Vmm &tmp_vmm, const rhs_address_t &rhs_addr,
        size_t group_stride) const {
    assert(is_data_supported(isa, data_type) && "unsupported data type");
    const auto &base = rhs_addr.getBase();
    const auto &v_idx = rhs_arg_static_params_.gather_idx_vmm;
    const auto &v_stage = rhs_arg_static_params_.narrow_stage_vmm;

    // The accumulator computes at e32/acc_lmul. Narrow dtypes briefly switch SEW
    // to the LMUL that shares that VLMAX (see the helpers above), so `vsetvli
    // x0,x0` keeps vl; they load into the staging group and widen into tmp_vmm
    // at e32/acc_lmul. f32/s32 load at the accumulator's active e32/acc_lmul.
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            if (rhs_addr.isGather())
                host_->vluxei32_v(tmp_vmm, base, v_idx);
            else if (rhs_addr.isStrided())
                host_->vlse32_v(tmp_vmm, base, rhs_addr.getStride());
            else
                host_->vle32_v(tmp_vmm, base);
            break;
        case data_type::s8:
        case data_type::u8:
            host_->vsetvli(x0, x0, SEW::e8, rhs_e8_lmul(group_stride), VTA::ta,
                    VMA::ma);
            if (rhs_addr.isGather())
                host_->vluxei32_v(v_stage, base, v_idx);
            else if (rhs_addr.isStrided())
                host_->vlse8_v(v_stage, base, rhs_addr.getStride());
            else
                host_->vle8_v(v_stage, base);
            host_->vsetvli(x0, x0, SEW::e32, acc_e32_lmul(group_stride),
                    VTA::ta, VMA::ma);
            if (data_type == data_type::s8)
                host_->vsext_vf4(tmp_vmm, v_stage);
            else
                host_->vzext_vf4(tmp_vmm, v_stage);
            break;
        case data_type::f16:
        case data_type::bf16:
            host_->vsetvli(x0, x0, SEW::e16, rhs_e16_lmul(group_stride),
                    VTA::ta, VMA::ma);
            if (rhs_addr.isBroadcast())
                host_->vlse16_v(v_stage, base, x0); // stride-0 broadcast
            else if (rhs_addr.isGather())
                host_->vluxei32_v(v_stage, base, v_idx);
            else if (rhs_addr.isStrided())
                host_->vlse16_v(v_stage, base, rhs_addr.getStride());
            else
                host_->vle16_v(v_stage, base);
            if (data_type == data_type::bf16)
                host_->vfwcvtbf16_f_f_v(tmp_vmm, v_stage);
            else
                host_->vfwcvt_f_f_v(
                        tmp_vmm, v_stage); // e16/(acc_lmul/2) -> e32/acc
            host_->vsetvli(x0, x0, SEW::e32, acc_e32_lmul(group_stride),
                    VTA::ta, VMA::ma);
            break;
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::cvt_to_f32(const Vmm &tmp_vmm) const {
    // Signed convert is exact for u8 too (vzext leaves 0..255, below 2^31; the
    // same reasoning as x64's shared uni_vcvtdq2ps).
    host_->vfcvt_f_x_v(tmp_vmm, tmp_vmm);
}

template class jit_uni_binary_injector_t<v>;
template class jit_uni_binary_injector_t<zvfh>;
template class jit_uni_binary_injector_t<zvfbfwma>;

#undef VCHECK_BIN_INJ_BOOL

} // namespace binary_injector
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
