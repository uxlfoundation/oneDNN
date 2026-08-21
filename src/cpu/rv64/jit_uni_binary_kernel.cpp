/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <cstring>

#include "cpu/rv64/jit_uni_binary_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

#define PARAM_OFF(x) offsetof(jit_uni_binary_args_t, x)

// Keep in sync with the pd's set in jit_uni_binary.cpp (x64 duplicates the
// file-local function the same way): the codegen-time classification must
// match the pd's, and both stay the x64 subset of the injector's abilities.
static bcast_set_t get_supported_postops_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::no_broadcast};
}

jit_uni_binary_kernel_t::jit_uni_binary_kernel_t(
        const binary_pd_t *pd, const jit_binary_conf_t &conf, bool tail_kernel)
    // rv64's DECLARE_CPU_JIT_AUX_FUNCTIONS has no static jit_name(); keep the
    // stable generated-code name registered with jit_utils.
    : binary_kernel_t(pd, conf, "jit_uni_binary", tail_kernel) {
    init();
}

void jit_uni_binary_kernel_t::init() {
    if (conf_.with_postops) init_post_ops_injector();
}

void jit_uni_binary_kernel_t::init_post_ops_injector() {
    const auto &po = pd_->attr()->post_ops_;
    const memory_desc_wrapper dst_d(pd_->dst_md(0));

    // The eltwise post-op injector gets five aux groups (v8/v12/v16 +
    // v24/v28); v24/v28 are free at m4 (accumulator v4, src1 v8, staging
    // v12/v16, binary rhs v20), so the heavy eltwise algs (log/soft_relu/
    // gelu_erf) fit here too.
    eltwise_injector::static_params_t esp(vreg_src1_, vreg_tmp0_, vreg_tmp1_,
            VReg(24), VReg(28), fa0, fa1, reg_tmp_, /*is_fwd=*/true);
    // Binary rhs addressing: the injector derives each entry's broadcast
    // strategy and rhs dtype from post_op + dst_d at call time and recovers
    // the output element offset from the dst chunk address and dst_orig (x64
    // model). reg_tmp1_/reg_bytes_ are its address scratch, v24 the per-oc
    // gather index (the eltwise aux3 shares it but runs in a different
    // post-op entry, so there is no temporal overlap), and v12 the narrow-rhs
    // staging (distinct from v20/v24 so the widen is a legal overlap). A
    // select post-op consumes the v0 mask plus the shared v20/v12 helpers; v0
    // is dead here between injector calls (the main op's mask is consumed
    // before post-ops run).
    // x64-style rhs_arg_static_params_t: helper vmm index, rhs addr/helper/
    // cache GPRs, preserve flags (GPR preservation shields the injector's
    // fixed X_TMP scratch, which doubles as this kernel's t4/t5 loop state;
    // vmm helpers are host-reserved statically -> false), the args-struct
    // offsets of the rhs pointer array and dst_orig, dst_d.
    binary_injector::rhs_arg_static_params_t rhs_arg_bsp(vreg_rhs_.getIdx(),
            reg_tmp1_, reg_bytes_, reg_tmp_, true /*preserve gpr*/,
            false /*preserve vmm*/, PARAM_OFF(post_ops_binary_rhs_arg_vec),
            PARAM_OFF(dst_orig), dst_d);
    rhs_arg_bsp.rhs_dt_helper_freg = fa7;
    rhs_arg_bsp.gather_idx_vmm = VReg(24);
    rhs_arg_bsp.narrow_stage_vmm = vreg_tmp0_;
    // Channel-aligned per-oc rhs (x64's per-call scalar addressing): the
    // dispatch routes per-oc post-op cases to the per_c/per_w strategies, so
    // the injector recovers the channel offset from the dst address and loads
    // the rhs contiguously (or broadcasts one value) instead of the per-lane
    // gather. op_t::none cannot be routed and keeps the gather (the dispatch
    // predicate must equal this creation predicate).
    rhs_arg_bsp.per_oc_lanes_are_channels
            = conf_.postops_per_oc_broadcast_exists
            && conf_.op_type != op_t::none;
    const binary_injector::static_params_t bsp(
            reg_param_, get_supported_postops_bcast_strategies(), rhs_arg_bsp);

    // The in-kernel sum is a host lambda (the x64/aarch64 lambda-injector
    // scheme): the postops injector invokes it at the sum's exact chain
    // position, so sums land anywhere in the chain. Following x64's do_sum
    // definition, a zero sum scale drops the old-dst read (no lambda).
    injector::lambda_jit_injectors_t lambda_jit_injectors;
    if (conf_.do_sum)
        lambda_jit_injectors.emplace(primitive_kind::sum, [this]() {
            // old dst -> vreg_tmp1_ (f32)
            load_vector(conf_.dst_type, vreg_tmp1_, vreg_tmp0_, reg_dst_);
            vfmacc_vf(vreg_src0_, freg_sum_scale_, vreg_tmp1_);
        });

    // Instantiate the injector at the isa the pd validated the post-op chain
    // with so its is_data_supported contract admits exactly the rhs dtypes the
    // pd accepted. RVV codegen is otherwise identical for each instantiation.
    if (mayiuse(zvfbfwma))
        postops_injector_zvfbfwma_.reset(
                new injector::jit_uni_postops_injector_t<zvfbfwma>(
                        this, po, bsp, esp, lambda_jit_injectors));
    else if (mayiuse(zvfh))
        postops_injector_zvfh_.reset(
                new injector::jit_uni_postops_injector_t<zvfh>(
                        this, po, bsp, esp, lambda_jit_injectors));
    else
        postops_injector_v_.reset(new injector::jit_uni_postops_injector_t<v>(
                this, po, bsp, esp, lambda_jit_injectors));
}

void jit_uni_binary_kernel_t::apply_postops() {
    // The per-register rhs addressing (x64's rhs_arg_params analog): map the
    // accumulator to the register holding its dst chunk address; the injector
    // recovers the element offset via dst_orig.
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    if (conf_.with_binary)
        rhs_arg_params.vmm_idx_to_out_reg.emplace(
                vreg_src0_.getIdx(), reg_dst_);

    // The kernel computes at e32/m4 (group_stride 4); the injector uses it to
    // load a narrow rhs at the matching LMUL.
    if (postops_injector_zvfbfwma_)
        postops_injector_zvfbfwma_->compute_vector(
                vreg_src0_.getIdx(), rhs_arg_params, compute_group_stride_);
    else if (postops_injector_zvfh_)
        postops_injector_zvfh_->compute_vector(
                vreg_src0_.getIdx(), rhs_arg_params, compute_group_stride_);
    else
        postops_injector_v_->compute_vector(
                vreg_src0_.getIdx(), rhs_arg_params, compute_group_stride_);
}

void jit_uni_binary_kernel_t::load_kernel_params() {
    ld(reg_src0_, reg_param_, PARAM_OFF(src0));
    ld(reg_src1_, reg_param_, PARAM_OFF(src1));
    if (conf_.is_ternary_op) ld(reg_src2_, reg_param_, PARAM_OFF(src2));
    ld(reg_dst_, reg_param_, PARAM_OFF(dst));
    ld(reg_work_amount_, reg_param_, PARAM_OFF(work_amount));
    if (conf_.do_scale_src0)
        flw(freg_scales_src0_, reg_param_, PARAM_OFF(scales_src0));
    if (conf_.do_scale_src1)
        flw(freg_scales_src1_, reg_param_, PARAM_OFF(scales_src1));
    if (conf_.do_sum) flw(freg_sum_scale_, reg_param_, PARAM_OFF(sum_scale));
}

void jit_uni_binary_kernel_t::materialize_cmp() {
    // Comparisons yield the oneDNN-required f32 0.0/1.0 (x64 blends from a
    // preloaded ones register under the compare mask instead).
    const FReg freg_zero = fa0, freg_one = fa1;
    fmv_w_x(freg_zero, x0);
    li(reg_tmp_, 0x3f800000);
    fmv_w_x(freg_one, reg_tmp_);
    vfmv_v_f(vreg_src0_, freg_zero);
    // vreg_src0_[i] = v0[i] ? 1.0 : 0.0
    vfmerge_vfm(vreg_src0_, vreg_src0_, freg_one);
}

void jit_uni_binary_kernel_t::perform_op(const VReg &v0, const VReg &v1) {
    using namespace alg_kind;
    switch (pd_->desc()->alg_kind) {
        case binary_add: vfadd_vv(v0, v0, v1); break;
        case binary_sub: vfsub_vv(v0, v0, v1); break;
        case binary_mul: vfmul_vv(v0, v0, v1); break;
        case binary_div: vfdiv_vv(v0, v0, v1); break;
        case binary_max:
            // nstl::max(src0,src1) = (src1 < src0) ? src0 : src1 (picks src1
            // on ties/unordered, matching the reference and x86 vmaxps).
            vmflt_vv(vreg_mask_, v1, v0);
            vmerge_vvm(v0, v1, v0);
            break;
        case binary_min:
            // nstl::min(src0,src1) = (src0 < src1) ? src0 : src1.
            vmflt_vv(vreg_mask_, v0, v1);
            vmerge_vvm(v0, v1, v0);
            break;
        // vmfgt/vmfge have no vv form: swap operands.
        case binary_ge:
            vmfle_vv(vreg_mask_, v1, v0);
            materialize_cmp();
            break;
        case binary_gt:
            vmflt_vv(vreg_mask_, v1, v0);
            materialize_cmp();
            break;
        case binary_le:
            vmfle_vv(vreg_mask_, v0, v1);
            materialize_cmp();
            break;
        case binary_lt:
            vmflt_vv(vreg_mask_, v0, v1);
            materialize_cmp();
            break;
        case binary_eq:
            vmfeq_vv(vreg_mask_, v0, v1);
            materialize_cmp();
            break;
        case binary_ne:
            vmfne_vv(vreg_mask_, v0, v1);
            materialize_cmp();
            break;
        default: assert(!"unsupported operation!"); break;
    }
}

void jit_uni_binary_kernel_t::perform_op(const VReg &v0, const FReg &s1) {
    using namespace alg_kind;
    switch (pd_->desc()->alg_kind) {
        case binary_add: vfadd_vf(v0, v0, s1); break;
        case binary_sub: vfsub_vf(v0, v0, s1); break;
        case binary_mul: vfmul_vf(v0, v0, s1); break;
        case binary_div: vfdiv_vf(v0, v0, s1); break;
        case binary_max:
            // nstl::max(src0,src1) = (src1 < src0) ? src0 : src1 (picks src1
            // on ties/unordered, matching the reference and x86 vmaxps).
            vfmv_v_f(vreg_src1_, s1);
            vmflt_vv(vreg_mask_, vreg_src1_, v0);
            vmerge_vvm(v0, vreg_src1_, v0);
            break;
        case binary_min:
            // nstl::min(src0,src1) = (src0 < src1) ? src0 : src1.
            vfmv_v_f(vreg_src1_, s1);
            vmflt_vv(vreg_mask_, v0, vreg_src1_);
            vmerge_vvm(v0, vreg_src1_, v0);
            break;
        case binary_ge:
            vmfge_vf(vreg_mask_, v0, s1);
            materialize_cmp();
            break;
        case binary_gt:
            vmfgt_vf(vreg_mask_, v0, s1);
            materialize_cmp();
            break;
        case binary_le:
            vmfle_vf(vreg_mask_, v0, s1);
            materialize_cmp();
            break;
        case binary_lt:
            vmflt_vf(vreg_mask_, v0, s1);
            materialize_cmp();
            break;
        case binary_eq:
            vmfeq_vf(vreg_mask_, v0, s1);
            materialize_cmp();
            break;
        case binary_ne:
            vmfne_vf(vreg_mask_, v0, s1);
            materialize_cmp();
            break;
        default: assert(!"unsupported operation!"); break;
    }
}

void jit_uni_binary_kernel_t::perform_ternary_op(
        const VReg &v0, const VReg &v1, const VReg &vreg_stage) {
    // dst = src2 ? src0 : src1. Materialize a scalar-broadcast src1, then set
    // v0 = (src2 == 0) and merge (v0 ? src1 : src0). The s8 condition is
    // forced by the common binary descriptor; the mask stays valid across the
    // SEW switch back to the compute vtype (same vl).
    if (conf_.broadcast_src1_value) vfmv_v_f(v1, freg_bcast_src1_);
    vsetvli(x0, reg_vl_, SEW::e8, stage_e8_lmul_, VTA::ta, VMA::ma);
    vle8_v(vreg_stage, reg_src2_);
    vmseq_vx(vreg_mask_, vreg_stage, x0); // v0 = (src2 == 0)
    to_compute_vtype();
    vmerge_vvm(v0, v0, v1);
}

void jit_uni_binary_kernel_t::compute_bcast() {
    // Preload the broadcast scalar src1 (f32) once per kernel call, scaled
    // once (x64 compute_bcast broadcasts into a vector register; RVV keeps a
    // scalar FP register and uses the .vf instruction forms).
    if (conf_.broadcast_src1_value) {
        load_scalar(conf_.src1_type, freg_bcast_src1_, reg_src1_);
        if (conf_.do_scale_src1)
            fmul_s(freg_bcast_src1_, freg_bcast_src1_, freg_scales_src1_);
    }
}

void jit_uni_binary_kernel_t::load_vector(data_type_t dt, const VReg &vreg,
        const VReg &vreg_stage, const Reg &reg_ptr, dim_t stride_elems) {
    using namespace data_type;
    // The src1 byte stride is hoisted into reg_src1_stride_ by
    // forward_over_outer_dims (the only strided consumer).
    const bool strided = stride_elems > 1;
    if (dt == f32) {
        to_compute_vtype();
        strided ? vlse32_v(vreg, reg_ptr, reg_src1_stride_)
                : vle32_v(vreg, reg_ptr);
    } else if (dt == s32) {
        to_compute_vtype();
        strided ? vlse32_v(vreg, reg_ptr, reg_src1_stride_)
                : vle32_v(vreg, reg_ptr);
        vfcvt_f_x_v(vreg, vreg);
    } else if (dt == f16 || dt == bf16) {
        vsetvli(x0, reg_vl_, SEW::e16, stage_e16_lmul_, VTA::ta, VMA::ma);
        strided ? vlse16_v(vreg_stage, reg_ptr, reg_src1_stride_)
                : vle16_v(vreg_stage, reg_ptr);
        if (dt == bf16)
            vfwcvtbf16_f_f_v(vreg, vreg_stage);
        else
            vfwcvt_f_f_v(vreg, vreg_stage); // e16m2 -> e32m4
        to_compute_vtype();
    } else { // s8 / u8
        // vsext/vzext.vf4 read the source group as 8-bit at the e32/m4 dest
        // vtype, so load at e8/m1 then switch vtype before extending.
        vsetvli(x0, reg_vl_, SEW::e8, stage_e8_lmul_, VTA::ta, VMA::ma);
        strided ? vlse8_v(vreg_stage, reg_ptr, reg_src1_stride_)
                : vle8_v(vreg_stage, reg_ptr);
        to_compute_vtype();
        if (dt == s8) {
            vsext_vf4(vreg, vreg_stage);
            vfcvt_f_x_v(vreg, vreg);
        } else {
            vzext_vf4(vreg, vreg_stage);
            vfcvt_f_xu_v(vreg, vreg);
        }
    }
}

void jit_uni_binary_kernel_t::load_scalar(
        data_type_t dt, const FReg &freg, const Reg &reg_ptr) {
    using namespace data_type;
    if (dt == f32) {
        flw(freg, reg_ptr, 0);
    } else if (dt == s32) {
        lw(reg_tmp_, reg_ptr, 0);
        fcvt_s_w(freg, reg_tmp_);
    } else if (dt == s8) {
        lb(reg_tmp_, reg_ptr, 0);
        fcvt_s_w(freg, reg_tmp_);
    } else if (dt == u8) {
        lbu(reg_tmp_, reg_ptr, 0);
        fcvt_s_wu(freg, reg_tmp_);
    } else { // f16/bf16: widen via the vector unit, extract lane 0
        // Single-lane extraction (AVL = 1), so these LMULs are unrelated to the
        // compute vtype and stay literal: any e16/e32 pair works at vl = 1.
        li(reg_tmp_, 1);
        vsetvli(x0, reg_tmp_, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        vle16_v(vreg_rhs_, reg_ptr);
        if (dt == bf16)
            vfwcvtbf16_f_f_v(vreg_src1_, vreg_rhs_);
        else
            vfwcvt_f_f_v(vreg_src1_, vreg_rhs_); // e16m1 -> e32m2
        vsetvli(x0, reg_tmp_, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        vfmv_f_s(freg, vreg_src1_);
    }
}

void jit_uni_binary_kernel_t::store_vector(data_type_t dt, const VReg &vreg,
        const VReg &vreg_stage, const Reg &reg_ptr) {
    using namespace data_type;
    if (dt == f32) {
        to_compute_vtype();
        vse32_v(vreg, reg_ptr);
    } else if (dt == s32) {
        to_compute_vtype();
        load_f32_const(freg_tmp_, -2147483648.0f);
        vfmax_vf(vreg, vreg, freg_tmp_);
        // RISC-V vfcvt.x.f saturates; max f32 below 2^31 is 2147483520.
        load_f32_const(freg_tmp_, 2147483520.0f);
        vfmin_vf(vreg, vreg, freg_tmp_);
        vfcvt_x_f_v(vreg, vreg);
        vse32_v(vreg, reg_ptr);
    } else if (dt == f16 || dt == bf16) {
        vsetvli(x0, reg_vl_, SEW::e16, stage_e16_lmul_, VTA::ta, VMA::ma);
        if (dt == bf16)
            vfncvtbf16_f_f_w(vreg_stage, vreg);
        else
            vfncvt_f_f_w(vreg_stage, vreg); // e32m4 -> e16m2
        vse16_v(vreg_stage, reg_ptr);
    } else { // s8 / u8
        const bool is_s8 = dt == s8;
        to_compute_vtype();
        load_f32_const(freg_tmp_, is_s8 ? -128.0f : 0.0f);
        vfmax_vf(vreg, vreg, freg_tmp_);
        load_f32_const(freg_tmp_, is_s8 ? 127.0f : 255.0f);
        vfmin_vf(vreg, vreg, freg_tmp_);
        if (is_s8)
            vfcvt_x_f_v(vreg, vreg);
        else
            vfcvt_xu_f_v(vreg, vreg);
        // Narrow i32(m4) -> i16(m2) -> i8(m1) through the staging group; values
        // are pre-clamped so the vnsrl-by-0 truncation is exact.
        vsetvli(x0, reg_vl_, SEW::e16, stage_e16_lmul_, VTA::ta, VMA::ma);
        vnsrl_wi(vreg_stage, vreg, 0);
        vsetvli(x0, reg_vl_, SEW::e8, stage_e8_lmul_, VTA::ta, VMA::ma);
        vnsrl_wi(vreg_stage, vreg_stage, 0);
        vse8_v(vreg_stage, reg_ptr);
    }
}

void jit_uni_binary_kernel_t::compute_dst_body() {
    // src0 -> vreg_src0_ (f32)
    load_vector(conf_.src0_type, vreg_src0_, vreg_tmp0_, reg_src0_);
    if (conf_.do_scale_src0)
        vfmul_vf(vreg_src0_, vreg_src0_, freg_scales_src0_);
    if (!conf_.broadcast_src1_value) {
        // src1 -> vreg_src1_ (f32); strided for different plain layouts.
        load_vector(conf_.src1_type, vreg_src1_, vreg_tmp1_, reg_src1_,
                conf_.src1_stride);
        if (conf_.do_scale_src1)
            vfmul_vf(vreg_src1_, vreg_src1_, freg_scales_src1_);
    }

    if (conf_.is_ternary_op)
        perform_ternary_op(vreg_src0_, vreg_src1_, vreg_tmp1_);
    else if (conf_.broadcast_src1_value)
        perform_op(vreg_src0_, freg_bcast_src1_);
    else
        perform_op(vreg_src0_, vreg_src1_);
}

void jit_uni_binary_kernel_t::compute_dst() {
    compute_dst_body();
    if (has_postops_injector()) apply_postops();
    store_vector(conf_.dst_type, vreg_src0_, vreg_tmp1_, reg_dst_);
}

void jit_uni_binary_kernel_t::forward() {
    compute_bcast(); // bcast scalar loaded just one time per a kernel call

    // Channel-aligned per-oc post-op rhs addressing (see conf /
    // init_post_ops_injector).
    const bool per_oc_rhs = conf_.postops_per_oc_broadcast_exists
            && conf_.op_type != op_t::none;

    if (conf_.is_src_different_layouts) {
        forward_over_outer_dims();
        return;
    }

    const bool is_blk = conf_.op_type == op_t::c_blocked;
    const memory_desc_wrapper dst_d(pd_->dst_md(0));
    const int c_blk = is_blk ? (int)dst_d.blocking_desc().inner_blks[0] : 0;
    // per_c x c_blocked: src1 is neither broadcast nor advancing -- the same
    // c_block vector is re-read every block (x64's offt_src1_ == 0).
    const bool src1_block_reuse
            = is_blk && !conf_.broadcast_src1_value && !conf_.use_stride_src1;
    // Step one channel block per iteration (AVL = a constant the vsetvli
    // always grants in full, VLMAX(e32, m4) >= 16 >= c_block): the tail
    // kernel must stop at the valid lanes, src1 reuse must not cross a block,
    // and the per-oc rhs slice must stay within one block. The driver only
    // ever sends multiples of c_block. The plain-shape/no-constraint cases
    // keep the full-VLMAX VLA loop.
    const bool block_stepped
            = is_tail_kernel_ || src1_block_reuse || (is_blk && per_oc_rhs);
    if (block_stepped)
        li(reg_blk_avl_, is_tail_kernel_ ? (uint32_t)tail_size_ : c_blk);

    Label loop, end;
    L(loop);
    beqz(reg_work_amount_, end);

    if (block_stepped)
        vsetvli(reg_vl_, reg_blk_avl_, SEW::e32, compute_lmul_, VTA::ta,
                VMA::ma);
    else
        set_compute_vl();
    compute_dst();
    if (is_tail_kernel_) zero_pad_tail(c_blk);

    // Advance pointers by vl elements using the vl granted by vsetvli, not
    // the requested work amount.
    auto advance = [&](const Reg &reg_ptr, data_type_t dt) {
        const int dt_size = (int)types::data_type_size(dt);
        const int shift = dt_size == 4 ? 2 : (dt_size == 2 ? 1 : 0);
        if (shift)
            slli(reg_bytes_, reg_vl_, shift);
        else
            mv(reg_bytes_, reg_vl_);
        add(reg_ptr, reg_ptr, reg_bytes_);
    };
    // Block-stepped: advance by the constant c_block regardless of vl (the
    // tail kernel computes only the valid lanes but still steps whole
    // blocks).
    auto advance_const = [&](const Reg &reg_ptr, int nelems, data_type_t dt) {
        addi(reg_ptr, reg_ptr, nelems * (int)types::data_type_size(dt));
    };
    if (block_stepped) {
        advance_const(reg_src0_, c_blk, conf_.src0_type);
        advance_const(reg_dst_, c_blk, conf_.dst_type);
        if (!conf_.broadcast_src1_value && !src1_block_reuse)
            advance_const(reg_src1_, c_blk, conf_.src1_type);
        if (conf_.is_ternary_op) advance_const(reg_src2_, c_blk, data_type::s8);
        addi(reg_work_amount_, reg_work_amount_, -c_blk);
    } else {
        advance(reg_src0_, conf_.src0_type);
        advance(reg_dst_, conf_.dst_type);
        if (!conf_.broadcast_src1_value) advance(reg_src1_, conf_.src1_type);
        // select condition: full (dst-shaped), s8 (1 B/elem)
        if (conf_.is_ternary_op) advance(reg_src2_, data_type::s8);
        sub(reg_work_amount_, reg_work_amount_, reg_vl_);
    }
    j_(loop);
    L(end);
}

void jit_uni_binary_kernel_t::forward_over_outer_dims() {
    // x64 iterates the gathered src1 through reg_src1_stride_range_ and, when
    // a run of outer_dims elements completes, rebases src1 to the next
    // contiguous element; outer_dims and src1_stride are codegen constants
    // here, so the stride and the per-run rebase are baked instead of the
    // runtime indices vector + stride-range register.
    const int es1 = (int)types::data_type_size(conf_.src1_type);
    load_imm64(reg_src1_stride_, (conf_.src1_stride * es1));

    Label run_loop, inner_loop, run_avl_ready, end;
    L(run_loop);
    beqz(reg_work_amount_, end);
    load_imm64(reg_blk_avl_, conf_.outer_dims); // remaining in this run
    bge(reg_work_amount_, reg_blk_avl_, run_avl_ready);
    mv(reg_blk_avl_, reg_work_amount_);
    L(run_avl_ready);
    L(inner_loop);
    vsetvli(reg_vl_, reg_blk_avl_, SEW::e32, compute_lmul_, VTA::ta, VMA::ma);
    compute_dst();

    auto advance = [&](const Reg &reg_ptr, data_type_t dt) {
        const int dt_size = (int)types::data_type_size(dt);
        const int shift = dt_size == 4 ? 2 : (dt_size == 2 ? 1 : 0);
        if (shift)
            slli(reg_bytes_, reg_vl_, shift);
        else
            mv(reg_bytes_, reg_vl_);
        add(reg_ptr, reg_ptr, reg_bytes_);
    };
    advance(reg_src0_, conf_.src0_type);
    advance(reg_dst_, conf_.dst_type);
    // strided src1: advance by vl * stride * sizeof(dt).
    mul(reg_bytes_, reg_vl_, reg_src1_stride_);
    add(reg_src1_, reg_src1_, reg_bytes_);
    if (conf_.is_ternary_op) advance(reg_src2_, data_type::s8);
    sub(reg_blk_avl_, reg_blk_avl_, reg_vl_);
    sub(reg_work_amount_, reg_work_amount_, reg_vl_);
    bnez(reg_blk_avl_, inner_loop);
    // Run done: rebase src1 to the next contiguous element,
    // -(outer_dims * stride - 1) * es1 (x64 pops the saved base + 1 elem).
    load_imm64(reg_bytes_, (conf_.outer_dims * conf_.src1_stride - 1) * es1);
    sub(reg_src1_, reg_src1_, reg_bytes_);
    j_(run_loop);
    L(end);
}

void jit_uni_binary_kernel_t::generate() {
    // rv64 jit_generator_t has no preamble()/postamble(): the kernel is a
    // leaf function touching only caller-saved registers (x64 additionally
    // emits the eltwise constant table here; RVV loads constants inline).
    load_kernel_params();
    forward();
    ret();
}

// --- rv64-specific low-level helpers (no x64 method analog; cf. aarch64,
// which defines its arch-specific helpers after generate()) ---

void jit_uni_binary_kernel_t::set_compute_vl() {
    vsetvli(reg_vl_, reg_work_amount_, SEW::e32, compute_lmul_, VTA::ta,
            VMA::ma);
}

void jit_uni_binary_kernel_t::to_compute_vtype() {
    vsetvli(x0, reg_vl_, SEW::e32, compute_lmul_, VTA::ta, VMA::ma);
}

void jit_uni_binary_kernel_t::load_f32_const(const FReg &freg, float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    li(reg_tmp_, bits);
    fmv_w_x(freg, reg_tmp_);
}

void jit_uni_binary_kernel_t::zero_pad_tail(int c_blk) {
    // Zero the padded lanes [tail, c_block) of the current block (x64's
    // tail-kernel zero-padding store); zero BYTES at e8 cover any dst dtype.
    const int dsz = (int)types::data_type_size(conf_.dst_type);
    const int pad_bytes = (c_blk - (int)tail_size_) * dsz;
    load_imm64(reg_tmp_, pad_bytes);
    // pad_bytes <= 15 * 4 = 60 <= VLMAX(e8, m4) = VLEN / 2 for VLEN >= 128.
    vsetvli(x0, reg_tmp_, SEW::e8, compute_lmul_, VTA::ta, VMA::ma);
    vmv_v_x(vreg_tmp1_, x0);
    addi(reg_bytes_, reg_dst_, (int)tail_size_ * dsz);
    vse8_v(vreg_tmp1_, reg_bytes_);
}

#undef PARAM_OFF

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
