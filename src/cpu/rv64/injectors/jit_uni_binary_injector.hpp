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

#ifndef CPU_RV64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP
#define CPU_RV64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP

#include <functional>
#include <map>
#include <utility>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "cpu/binary_injector_utils.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/injectors/injector_utils.hpp"
#include "cpu/rv64/jit_generator.hpp"

// This injector is a faithful port of the x64/aarch64 binary post-op injector
// (aarch64 is the primary reference: it ports x64 to a scalable-vector ISA, as
// SVE and RVV are both VLA). The file mirrors aarch64's structure
// function-by-function, variable-by-variable, and in declaration order. The
// intentional (A-class) differences from aarch64/x64 are: RV64 register
// conventions, RVV instruction selection, no bf16, the <v>/<zvfh> ISA template
// selecting only a registration slot, and no tail machinery (a single VLA
// vsetvli loop subsumes the tail, so the with_tail / tail-opmask / tail-load
// family is dropped). The addressing math (calculate_*_{ncsp,blocked,nspc,cspn})
// is arch-neutral scalar div/mod and is ported verbatim; the loads/compute and
// the per_oc VLA gather are the RVV-native realizations.

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace binary_injector {
using dnnl::impl::cpu::binary_injector_utils::get_src1_desc;
using dnnl::impl::cpu::binary_injector_utils::get_src2_desc;
using dnnl::impl::cpu::binary_injector_utils::prepare_binary_args;

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops);

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);
bool any_binary_postop_rhs_non_scalar_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);
bool any_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Represents params related to all binary post-ops right-hand side arguments
 * (arg1) that don't change during jit_uni_binary_injector_t object lifetime
 * and between compute_vector_range calls.
 *
 * @param rhs_dt_helper_vmm_idx - index of vmm helper used when loading data for
 * calculations. Names a host-reserved e32/acc_lmul group (same LMUL as the
 * processed accumulators): group_stride-aligned, not v0 (the live mask
 * register), whole group inside the register file, and disjoint from the
 * processed vmms. Unlike x64, an invalid hint is not adjusted but rejected at
 * injection time (JIT_ASSERT; in Release the throw fails kernel creation).
 * @param rhs_addr_reg - gpr register, used as the currently processed address of
 * rhs tensor slice. Data of rhs(arg1) for the binary operation is loaded from address
 * stored inside rhs_addr_reg.
 * @param rhs_helper_reg - gpr register used as helper for calculations during data
 * loading phase.
 * @param rhs_addr_cache_reg - gpr register used for caching part of calculated
 * offset, this register is always preserved.
 * @param preserve_gpr_helpers - determines whether gpr registers specified above
 * should be preserved (pushed to stack and poped back afterwords) between
 * compute_vector_range calls.
 * @param preserve_vmm_helper - determines whether vmm helper register specified
 * above should be preserved between compute_vector_range calls.
 * @param abi_param_offset - offset to rhs tensor from first binary post-op operation
 * specified by user from runtime structure passed to kernel as abi param 1.
 * @param dst_orig_offset - offset 0 to destination tensor
 * @param dst_d - descriptor of destination tensor (result after applying all post-ops
 * operations).
 *
 * Scalar scratch: like x64 (which clobbers rax/rdx/r8/r9 and shields them
 * with the register-preserve guard when preserve_gpr_helpers is set), the
 * injector clobbers the fixed GPRs t4/t5/t6/a6/a7 (the aarch64 X_TMP_0..4
 * analogs; see the cpp). Hosts either keep them dead across the injection or
 * pass preserve_gpr_helpers = true; they must not be chosen as
 * rhs_addr_reg/rhs_helper_reg/rhs_addr_cache_reg.
 *
 * Note on x64 parity here: x64 also preserves rax/rdx even when
 * preserve_gpr_helpers is false, because x86 `div`/`mul` clobber them
 * implicitly and are never declared to the host. On rv64 the clobber set is
 * declared (the X_TMP_* above) and `divu`/`remu` take explicit operands, so
 * preserve_gpr_helpers == false genuinely means "the host guarantees these are
 * dead" and nothing is spilled. Do not add unconditional preservation: it would
 * contradict the flag and cost a stack frame per injection.
 *
 * rhs_addr_reg must never be x0: the injector unconditionally loads the rhs
 * base into it. rhs_helper_reg must not be x0 for any non-scalar strategy (it
 * receives the computed offset). rhs_addr_cache_reg is never written by the
 * rv64 injector at all (it only appears in the preserve list), so x0 is
 * harmless there. Both requirements are checked in compute_vector_range() once
 * the actual broadcasting strategy is known -- not at construction, where the
 * supported-strategy set may legitimately be wider than what is used.
 *
 * rv64-only members (A-class; no aarch64 analog):
 * @param rhs_dt_helper_freg - FP scalar register that receives a scalar
 * (broadcast) rhs value; RVV .vf instruction forms consume it directly, so a
 * full-vector broadcast into rhs_dt_helper_vmm is unnecessary.
 * @param gather_idx_vmm - vector group receiving the per-lane byte index for the
 * per_oc VLA gather (vluxei32). A VLA run may span channel boundaries, so per_oc
 * cannot be a contiguous load as on aarch64; each lane resolves its own channel.
 * @param narrow_stage_vmm - staging group for a narrow (s8/u8/f16) rhs load,
 * widened into rhs_dt_helper_vmm (a widening op may not overlap its source low
 * part).
 *
 * The select (ternary) post-op needs no dedicated condition scratch: the
 * condition is loaded through the shared rhs helper and immediately consumed
 * into the v0 mask (rv64 cannot stack-spill a vector like x64's push_vmm, and
 * the mask register is the only state that survives the src1 load reusing the
 * same helper).
 *
 * aarch64 members with no rv64 analog (VLA / no stack frame): the tail machinery
 * (tail_size, tail_opmask, reg_tail_size, use_exact_tail_scalar_bcast) is
 * subsumed by vsetvli's runtime vl. preserve_gpr_helpers works as on x64 (the
 * guard spills the helper GPRs and the fixed scratch around the injection).
 * preserve_vmm_helper spills the rhs dt helper as a whole e32/acc_lmul group
 * with a runtime vlenb-sized frame (x64 pushes Vmm(vmm_hint) the same way); the
 * rv64-only gather_idx/narrow_stage scratch stays host-reserved, and x64's
 * hint-conflict fallback of borrowing Vmm(0) is unsound here (v0 is the
 * architectural mask the compare/select paths write), so hosts must supply
 * non-conflicting hints (validated at injection time; an invalid hint fails
 * kernel creation). rhs_addr_cache_reg caches no addresses on rv64 -- every
 * call rebuilds the address from param1 -- so it is carried only to be spilled
 * by the preserve guard.
 */
struct rhs_arg_static_params_t {
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_vmm_idx,
            const Xbyak_riscv::Reg &rhs_addr_reg,
            const Xbyak_riscv::Reg &rhs_helper_reg,
            const Xbyak_riscv::Reg &rhs_addr_cache_reg,
            bool preserve_gpr_helpers, bool preserve_vmm_helper,
            std::size_t abi_param_offset, const memory_desc_wrapper &dst_d);
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_vmm_idx,
            const Xbyak_riscv::Reg &rhs_addr_reg,
            const Xbyak_riscv::Reg &rhs_helper_reg,
            const Xbyak_riscv::Reg &rhs_addr_cache_reg,
            bool preserve_gpr_helpers, bool preserve_vmm_helper,
            std::size_t abi_param_offset, std::size_t dst_orig_offset,
            const memory_desc_wrapper &dst_d);

    bool is_dst_orig_set() const noexcept { return is_dst_orig_set_; }

    std::size_t rhs_dt_helper_vmm_idx;
    Xbyak_riscv::Reg rhs_addr_reg;
    Xbyak_riscv::Reg rhs_helper_reg;
    Xbyak_riscv::Reg rhs_addr_cache_reg;
    bool preserve_gpr_helpers;
    bool preserve_vmm_helper;
    std::size_t abi_param_offset;
    std::size_t dst_orig_offset;
    memory_desc_wrapper dst_d;

    // rv64-only scratch (see the struct comment).
    Xbyak_riscv::FReg rhs_dt_helper_freg = Xbyak_riscv::FReg(0);
    Xbyak_riscv::VReg gather_idx_vmm = Xbyak_riscv::VReg(0);
    Xbyak_riscv::VReg narrow_stage_vmm = Xbyak_riscv::VReg(0);
    // rv64-only: the host maintains a byte offset (pooling) instead of a dst
    // address; the injector then adds it straight to the rhs base.
    bool off_is_bytes = false;
    // rv64-only: a no_broadcast rhs whose lanes are rhs_stride bytes apart (a
    // VLA host may vectorize across a non-innermost dst dim, e.g. pooling ncsp
    // full-dst); the rhs is then loaded with vlse.
    Xbyak_riscv::Reg rhs_stride = Xbyak_riscv::Reg(0);
    bool is_strided = false;
    // rv64-only: the host guarantees channel-aligned vectors for a per_oc/
    // per_oc_spatial rhs -- every vector's lanes lie within one channel
    // row/block (contiguous channels; a single channel for per_oc_spatial).
    // The injector then recovers the channel offset with the x64/aarch64
    // scalar calculate_oc_* math (from the dst address and dst_orig) and
    // loads the rhs contiguously (per_oc) or broadcasts one value
    // (per_oc_spatial) instead of the per-lane gather.
    bool per_oc_lanes_are_channels = false;

private:
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_vmm_idx,
            const Xbyak_riscv::Reg &rhs_addr_reg,
            const Xbyak_riscv::Reg &rhs_helper_reg,
            const Xbyak_riscv::Reg &rhs_addr_cache_reg,
            bool preserve_gpr_helpers, bool preserve_vmm_helper,
            std::size_t abi_param_offset, std::size_t dst_orig_offset,
            const memory_desc_wrapper &dst_d, bool is_dst_orig_set);

    bool is_dst_orig_set_;
};

/*
 * Represents params required by jit_uni_binary_injector_t that don't change
 * during it's entire lifetime.
 *
 * @param param1 - register storing abi param1. At the moment of calling
 * compute_vector_range method can be different than the default one defined
 * inside jit_generator.
 * @param bcast_set_t supported_strategy_set - set allowing disabling particular
 * bcast strategies
 * @param rhs_arg_static_params - params related to all binary post-ops right-hand side
 * arguments that don't change during entire lifetime of jit_uni_binary_injector_t
 * object.
 */
struct static_params_t {
    static_params_t(const Xbyak_riscv::Reg &param1,
            const bcast_set_t &supported_strategy_set,
            const rhs_arg_static_params_t &rhs_arg_static_params);
    static_params_t(const Xbyak_riscv::Reg &param1,
            const rhs_arg_static_params_t &rhs_arg_static_params);

    Xbyak_riscv::Reg param1;
    const bcast_set_t supported_strategy_set;
    rhs_arg_static_params_t rhs_arg_static_params;
};

/*
 * Represents the address of the rhs tensor slice (the rv64 analog of x64's
 * Xbyak::Address / aarch64's rhs_address_t). RVV has no memory operands, so the
 * "address" is a base gpr plus an immediate offset plus the load mode the
 * strategy selected:
 *   isBroadcast_ - one element at base_+offt_ broadcast over all lanes;
 *   isGather_    - per-lane byte-index gather (vluxei32) via gather_idx_vmm
 *                  (rv64-only: the per_oc VLA realization);
 *   otherwise    - contiguous unit-stride load (vle).
 * aarch64's `bits_` (operand width) has no rv64 analog: the width comes from
 * the active vtype, so it is not carried here.
 */
struct rhs_address_t {
    Xbyak_riscv::Reg base_;
    int64_t offt_ = 0;
    bool isBroadcast_ = false;
    bool isGather_ = false;
    bool isStrided_ = false;
    Xbyak_riscv::Reg stride_ = Xbyak_riscv::Reg(0);

    rhs_address_t(const Xbyak_riscv::Reg &base, const int64_t offt = 0,
            bool isBroadcast = false, bool isGather = false,
            bool isStrided = false,
            const Xbyak_riscv::Reg &stride = Xbyak_riscv::Reg(0))
        : base_(base)
        , offt_(offt)
        , isBroadcast_(isBroadcast)
        , isGather_(isGather)
        , isStrided_(isStrided)
        , stride_(stride) {}

    bool operator==(const rhs_address_t &rhs) const {
        return base_.getIdx() == rhs.base_.getIdx() && offt_ == rhs.offt_
                && isBroadcast_ == rhs.isBroadcast_
                && isGather_ == rhs.isGather_ && isStrided_ == rhs.isStrided_;
    }
    bool operator!=(const rhs_address_t &rhs) const { return !operator==(rhs); }

    Xbyak_riscv::Reg getBase() const { return base_; }
    Xbyak_riscv::Reg getStride() const { return stride_; }
    bool isBroadcast() const { return isBroadcast_; }
    bool isGather() const { return isGather_; }
    bool isStrided() const { return isStrided_; }
};

/*
 * An offset operand: either a register index or an address (base+offt). Mirrors
 * aarch64's rhs_operand_t; used by the no_broadcast/oc/sp/w offset-from-operand
 * dynamic-params path.
 */
struct rhs_operand_t {
    bool isAddress_ = false;
    uint32_t idx_ = 0;
    rhs_address_t address_ {Xbyak_riscv::Reg(0)};

    bool operator==(const rhs_operand_t &rhs) const {
        if (isAddress_ != rhs.isAddress_) return false;
        return isAddress_ ? address_ == rhs.address_ : idx_ == rhs.idx_;
    }
    bool operator!=(const rhs_operand_t &rhs) const { return !operator==(rhs); }

    bool isBroadcast() const { return isAddress_ && address_.isBroadcast(); }
};

/*
 * Represents params passed to compute_vector_range method of
 * jit_uni_binary_injector_t that can be different for each call. Contains
 * configurable std::maps where key is vmm index and value is the destination
 * tensor slice location (address / register / element offset). This is utilized
 * by the broadcasting mechanism. Mirrors aarch64's rhs_arg_dynamic_params_t.
 */
struct rhs_arg_dynamic_params_t {
    std::map<int, rhs_address_t> vmm_idx_to_out_addr;
    std::map<int, Xbyak_riscv::Reg> vmm_idx_to_out_reg;

    std::map<int, rhs_address_t> vmm_idx_to_out_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_out_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_out_off_oprnd;

    std::map<int, rhs_address_t> vmm_idx_to_oc_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_oc_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_oc_off_oprnd;

    std::map<int, rhs_address_t> vmm_idx_to_sp_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_sp_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_sp_off_oprnd;

    std::map<int, rhs_address_t> vmm_idx_to_mb_w_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_mb_w_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_mb_w_off_oprnd;

    std::map<int, rhs_address_t> vmm_idx_to_w_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_w_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_w_off_oprnd;
};

/*
 * Checks if src1 data type is supported by binary injector.
 */
bool is_data_supported(cpu_isa_t isa, data_type_t data_type);

/*
 * Checks if broadcast of src1 is supported by binary injector.
 */
bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Checks if binary injection for given args is supported.
 */
bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Main mechanism responsible for injecting binary postops supporting isa: rvv
 * (v, zvfh -- the template parameter selects the registration slot only; RVV
 * code is vector-length-agnostic) as well as data types: f32, s32, u8, s8, f16.
 */
template <cpu_isa_t isa>
class jit_uni_binary_injector_t {
    using Vmm = typename jit_isa_traits_t<isa>::Vmm;

public:
    jit_uni_binary_injector_t(
            jit_generator_t *host, const static_params_t &static_params);

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * ordered set of vector registers' indexes. group_stride is each
     * accumulator's e32 LMUL (registers per accumulator, one of 1/2/4); it
     * drives the SEW/LMUL of the narrow-dtype rhs load so the widen lands in
     * e32/LMUL (see load_rhs) and validates the set geometry. Function loads the
     * appropriate rhs slice per the internally determined broadcast strategy and
     * rhs_arg_params.
     */
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            size_t group_stride) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * the accumulators occupying the half-open physical register span
     * <start_idx, end_idx). group_stride is the number of vector registers per
     * accumulator (its LMUL/EMUL, one of 1/2/4; m8 is unsupported -- the helper
     * scratch is m4, see make_vmm_group_set); the run's group bases are
     * enumerated by stepping it, so unlike x64 an m>1 run maps to its true
     * bases, not consecutive indices. It is mandatory: an x64
     * compute_vector_range(start, end) must be converted deliberately, never
     * copied as a silent stride-1 enumeration (see make_vmm_group_set). Loads
     * the appropriate rhs slice per the internally determined broadcast
     * strategy and rhs_arg_params.
     */
    void compute_vector_range(size_t start_idx, size_t end_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            size_t group_stride) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * a single vector register index. Function loads appropriate slice of rhs tensor
     * for computations based on internally determined broadcast strategy and information
     * about stored data in particular vmm described inside rhs_arg_params.
     */
    void compute_vector(size_t idx, std::size_t rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            size_t group_stride) const;

private:
    /*
     * Validates the host-supplied temporary vmm hint against the processed
     * set. x64 adjusts an invalid hint to a free single vmm (0 or
     * max_vmm_idx); the rv64 helper is a host-reserved e32/acc_lmul group
     * (group_stride) with no legal substitute, so an invalid hint fails kernel
     * creation instead (JIT_ASSERT). The whole set is tested (not just its
     * endpoints) so a legal sparse set is not rejected.
     */
    void validate_temp_vmm_hint(const injector_utils::vmm_index_set_t &vmm_idxs,
            size_t group_stride, int max_vmm_idx) const;
    /*
     * Validates the rv64-only rhs scratch groups (narrow_stage_vmm for a
     * narrow-dtype vector load, gather_idx_vmm for a per-oc/spatial/w gather)
     * this entry will use: each must not overlap a live accumulator or the m4
     * helper, and a narrow gather's index and stage must be mutually disjoint.
     * x64 has no analog (its helper adjustment resolves all temporaries).
     */
    void validate_rhs_scratch(const injector_utils::vmm_index_set_t &vmm_idxs,
            size_t group_stride, const dnnl_post_ops::entry_t &post_op,
            broadcasting_strategy_t rhs_broadcasting_strategy) const;
    /*
     * Taking into account rhs_broadcasting_strategy and information from user
     * about tensor slice (rhs_arg_params) stored in Vmm(vmm_idx) calculates
     * address of rhs tensor slice needed for binary operation and returns it.
     * is_ternary_input selects the select (ternary) condition src2 instead of
     * src1 (x64 additionally passes is_first for its address cache; rv64
     * reloads the base on every call, so only the operand selector remains).
     */
    rhs_address_t prepare_rhs_arg_addr(std::size_t vmm_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            const broadcasting_strategy_t rhs_broadcasting_strategy,
            bool is_ternary_input) const;
    /*
     * Loads data and applies particular binary operation.
     */
    void inject_binary(const dnnl_post_ops::entry_t &post_op, Vmm dst,
            const rhs_address_t &rhs_addr, size_t group_stride) const;

    /*
     * Loads data and applies binary operation that require ternary inputs.
     * The caller has already consumed the condition operand into the v0 mask
     * (v0 = (cond == 0)); x64 instead carries the condition in a spilled
     * temporary vmm.
     */
    void inject_binary_with_ternary_op(const dnnl_post_ops::entry_t &post_op,
            Vmm dst, const rhs_address_t &rhs_addr, size_t group_stride) const;

    /*
     * Helper functions responsible for preparing rhs tensor slice address.
     */
    void append_offset_from_operand(
            const std::map<int, rhs_operand_t> &vmm_idx_to_elem_operand_off,
            int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
            const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes) const;
    void append_offset_under_mem_addr(
            const std::map<int, rhs_address_t> &vmm_idx_to_elem_addr_off,
            int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
            const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes) const;
    void append_value_offset(
            const std::map<int, size_t> &vmm_idx_to_elem_val_off, int vmm_idx,
            const Xbyak_riscv::Reg &addr_reg,
            std::size_t elem_size_bytes) const;

    void append_no_broadcast_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
            const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes) const;
    void calculate_no_broadcast(rhs_address_t addr, std::size_t offset,
            const Xbyak_riscv::Reg &out_reg) const;

    void append_oc_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
            const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes,
            bool is_per_oc_spatial) const;
    void calculate_oc_ncsp(const dim_t *strides,
            const Xbyak_riscv::Reg &tmp_reg, const bool residue = false) const;
    void calculate_oc_blocked(
            const dim_t *strides, const Xbyak_riscv::Reg &tmp_reg) const;
    void calculate_oc_nspc(
            const dim_t *strides, const Xbyak_riscv::Reg &tmp_reg) const;
    void calculate_oc_cspn(
            const dim_t *strides, const Xbyak_riscv::Reg &tmp_reg) const;

    // per_mb_spatial / per_mb_w / per_w are realized entirely by the per-lane
    // gather built inside these append_* helpers, so unlike x64/aarch64 there
    // are no calculate_mb_sp_* / calculate_mb_w_* / calculate_w_* scalar
    // variants here. Do not reintroduce them without also extending the
    // register-preserve guard in compute_vector_range(): the scalar offset math
    // clobbers X_TMP_1..4, which the guard currently only saves on the
    // per_oc-with-channel-lanes path (that is the sole live X_TMP_1..4 user).
    void append_mb_sp_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
            const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes) const;

    void append_mb_w_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
            const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes) const;

    void append_w_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_riscv::Reg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_riscv::Reg &addr_reg,
            const Xbyak_riscv::Reg &tmp_reg, std::size_t elem_size_bytes) const;

    // The comparison predicate is the alg itself: RVV encodes it in the
    // instruction mnemonic (vmfge/vmfgt/vmfle/vmflt/vmfeq/vmfne), so there is
    // no x86 _cmp_* immediate to forward.
    void compute_cmp_mask(
            const Vmm &lhs, const Vmm &rhs, const alg_kind_t cmp_alg) const;
    void execute_cmp_binary(const Vmm &dst, const Vmm &lhs, const Vmm &rhs,
            const alg_kind_t cmp_alg) const;
    // T is Vmm (.vv forms) or Xbyak_riscv::FReg (.vf forms) -- the rv64
    // counterpart of x64's register-vs-memory-operand rhs duality. x64 can pass
    // both through one `uni_v*ps(dst, lhs, rhs)` because every x86 instruction
    // takes a unified Operand; on rv64 the two forms are different mnemonics
    // and FReg/VReg share no base class, so the dispatch is done with these
    // overload pairs (the project builds as C++11 -- no `if constexpr`).
    void emit_arith(alg_kind_t alg, const Vmm &dst, const Vmm &lhs,
            const Xbyak_riscv::FReg &rhs) const;
    void emit_arith(alg_kind_t alg, const Vmm &dst, const Vmm &lhs,
            const Vmm &rhs) const;
    // Materialize a scalar rhs into the helper group; a vector rhs is returned
    // as is. Used by the min/max/compare sequences, which need a vector operand.
    Vmm materialize_rhs(const Xbyak_riscv::FReg &rhs) const;
    Vmm materialize_rhs(const Vmm &rhs) const;
    template <typename T>
    void execute_binary(alg_kind_t binary_alg, const Vmm &dst, const Vmm &lhs,
            const T &rhs) const;

    /*
     * Used in scalar broadcast strategy, loading a single value of the given
     * data type into the scalar helper FP register (RVV .vf forms broadcast it
     * implicitly; f16 has no scalar path and is loaded as a stride-0 vector
     * through load_rhs instead).
     */
    void execute_broadcast(const data_type_t &data_type,
            const Xbyak_riscv::FReg &tmp_freg,
            const rhs_address_t &rhs_addr) const;
    /*
     * Loads vl elements of the rhs slice into tmp_vmm as f32. VLA folds x64's
     * tail/no_tail split; narrow dtypes (s8/u8/f16) load into narrow_stage_vmm
     * and widen into tmp_vmm. group_stride is the accumulator's e32 LMUL: the
     * narrow load uses the SEW/LMUL that shares its VLMAX (e16 at LMUL/2, e8 at
     * LMUL/4) so the widen preserves vl.
     */
    void load_rhs(const data_type_t &data_type, const Vmm &tmp_vmm,
            const rhs_address_t &rhs_addr, size_t group_stride) const;
    void cvt_to_f32(const Vmm &tmp_vmm) const;

    jit_generator_t *host_;
    const rhs_arg_static_params_t rhs_arg_static_params_;
    const Xbyak_riscv::Reg param1_;
    const bcast_set_t supported_strategy_set_;
};

} // namespace binary_injector
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
