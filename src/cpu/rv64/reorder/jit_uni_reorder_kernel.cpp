/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022-2025 Arm Ltd. and affiliates
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

#include <cassert>
#include <cstddef>
#include <cstring>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/reorder/jit_uni_reorder_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace tr {

using namespace Xbyak_riscv;

namespace {
inline uint32_t float_as_u32(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}
} // namespace

/* ----------------------------- kernel_t dispatch ----------------------------- */

status_t kernel_t::desc_init(
        kernel_t::desc_t &desc, const prb_t &prb, int ndims_ker_max) {
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims) return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0) ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32_t::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
        case 0: return new jit_uni_reorder_kernel_f32_t(desc);
        default: assert(!"unknown kernel id"); return nullptr;
    }
    return nullptr;
}

/* ----------------------------- boilerplate ----------------------------- */

jit_uni_reorder_kernel_f32_t::jit_uni_reorder_kernel_f32_t(const desc_t &desc)
    : kernel_t(desc), jit_generator_t("jit_uni_reorder_kernel_f32") {
    itype_sz_ = (int)types::data_type_size(prb_.itype);
    otype_sz_ = (int)types::data_type_size(prb_.otype);
    stype_sz_ = (int)sizeof(float);
    interim_f32_ = interim_f32_needed(prb_, compensation_needed_);
}

void jit_uni_reorder_kernel_f32_t::operator()(const call_param_t *c) const {
    jit_generator_t::operator()(c);
}

void jit_uni_reorder_kernel_f32_t::operator()(
        const tail_call_param_t *c) const {
    jit_generator_t::operator()(c);
}

status_t jit_uni_reorder_kernel_f32_t::create_kernel() {
    return jit_generator_t::create_kernel();
}

/* ----------------------------- feature gating ----------------------------- */

bool jit_uni_reorder_kernel_f32_t::interim_f32_needed(
        const prb_t &prb, bool compensation_needed) {
    // A bit-exact integer copy is only possible when src/dst types match and no
    // arithmetic (scales / zero-points / beta / compensation / scale-adjust) is
    // requested. Everything else is computed through f32.
    return prb.itype != prb.otype || prb.src_scale_type != scale_type_t::NONE
            || prb.dst_scale_type != scale_type_t::NONE || prb.beta != 0.f
            || prb.req_src_zp || prb.req_dst_zp || compensation_needed
            || prb.scale_adjust != 1.f;
}

bool jit_uni_reorder_kernel_f32_t::applicable(const prb_t &p) {
    const bool is_f16 = p.itype == data_type::f16 || p.otype == data_type::f16;

    // The RVV vector extension is not part of the RV64 baseline, so the kernel
    // (which emits RVV unconditionally) must only be used when it is available.
    bool ok = mayiuse(v) && p.ndims > 0 && p.ndims <= ndims_jit_loop_max + 1
            && utils::one_of(p.itype, data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8, data_type::bf16,
                    data_type::f16)
            && utils::one_of(p.otype, data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8, data_type::bf16,
                    data_type::f16)
            // f16 (de)conversion needs the Zvfh extension.
            && IMPLICATION(is_f16, mayiuse(zvfh)) && p.ioff == 0 && p.ooff == 0
            && utils::one_of(p.beta, 0.f, 1.f) && prb_has_small_strides(p);
    if (!ok) return false;

    // s32 -> s32 with any arithmetic (scales / zero-points / sum /
    // scale-adjust) must stay integer-exact: the kernel computes through f32,
    // which rounds s32 values beyond 2^24, diverging from the integer-exact
    // reference. Pure s32 -> s32 copy (no attrs) is unaffected (bit-exact).
    // Defer the rest to reference.
    if (p.itype == data_type::s32 && p.otype == data_type::s32
            && (p.req_src_zp || p.req_dst_zp
                    || p.src_scale_type != scale_type_t::NONE
                    || p.dst_scale_type != scale_type_t::NONE || p.beta != 0.f
                    || p.scale_adjust != 1.f))
        return false;

    // Tail / zero-padding: JIT only the common single-inner-tail shape, i.e. the
    // tail lives in the innermost (vectorized) node[0], its parent is either
    // empty or a driver dim (whose curr_data_chunks the driver fills, so the
    // kernel needs no cascading counter writes), the zero-pad region is
    // contiguous in the output (os == 1), and no other kernel/driver node
    // carries a tail (which would need skip/zeroing). Everything else -> ref.
    if (p.is_tail_present) {
        const auto &n0 = p.nodes[0];
        if (n0.tail_size == 0) return false;
        if (n0.is_zero_pad_needed && n0.os != 1) return false;
        if (!n0.is_parent_empty() && n0.parent_node_id < p.ndims) return false;
        for (int d = 1; d < p.ndims; ++d)
            if (p.nodes[d].tail_size > 0) return false;
        for (int d = p.ndims; d < p.full_ndims; ++d)
            if (p.nodes[d].tail_size > 0) return false;
    }

    return true;
}

/* ----------------------------- emission helpers ----------------------------- */

void jit_uni_reorder_kernel_f32_t::preamble() {
    const Reg saved[] = {s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11};
    const int n = (int)(sizeof(saved) / sizeof(saved[0]));
    addi(sp, sp, -frame_size_);
    for (int i = 0; i < n; ++i)
        sd(saved[i], sp, i * 8);
}

void jit_uni_reorder_kernel_f32_t::postamble() {
    const Reg saved[] = {s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11};
    const int n = (int)(sizeof(saved) / sizeof(saved[0]));
    for (int i = 0; i < n; ++i)
        ld(saved[i], sp, i * 8);
    addi(sp, sp, frame_size_);
    ret();
}

void jit_uni_reorder_kernel_f32_t::set_fimm(const FReg &f, float val) {
    li(reg_tmp0_, float_as_u32(val));
    fmv_w_x(f, reg_tmp0_);
}

// Loads `vl` elements of type `dt` from `addr` (innermost stride `stride_elems`)
// and converts them into f32 in `v` (vtype must be e32/m1 on entry & exit).
void jit_uni_reorder_kernel_f32_t::load_to_f32(const VReg &v, const Reg &addr,
        data_type_t dt, ptrdiff_t stride_elems) {
    const bool unit = stride_elems == 1;
    const int sz = (int)types::data_type_size(dt);

    if (dt == data_type::f32 || dt == data_type::s32) {
        if (unit)
            vle32_v(v, addr);
        else {
            li(reg_tmp1_, (uint32_t)(stride_elems * sz));
            vlse32_v(v, addr, reg_tmp1_);
        }
        if (dt == data_type::s32) vfcvt_f_x_v(v, v);
        return;
    }

    if (dt == data_type::f16 || dt == data_type::bf16) {
        vsetvli(x0, reg_vl_, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
        if (unit)
            vle16_v(vreg_stg_, addr);
        else {
            li(reg_tmp1_, (uint32_t)(stride_elems * sz));
            vlse16_v(vreg_stg_, addr, reg_tmp1_);
        }
        if (dt == data_type::f16) {
            vfwcvt_f_f_v(v, vreg_stg_); // e16/mf2 -> e32/m1
            vsetvli(x0, reg_vl_, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        } else {
            // bf16 -> f32 is a left shift by 16 of the zero-extended bits.
            vsetvli(x0, reg_vl_, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
            vzext_vf2(v, vreg_stg_);
            li(reg_tmp1_, 16);
            vsll_vx(v, v, reg_tmp1_);
        }
        return;
    }

    // s8 / u8: stage as 8-bit (e8/mf4), widen to e32/m1, convert to f32.
    vsetvli(x0, reg_vl_, SEW::e8, LMUL::mf4, VTA::ta, VMA::ma);
    if (unit)
        vle8_v(vreg_stg_, addr);
    else {
        li(reg_tmp1_, (uint32_t)(stride_elems * sz));
        vlse8_v(vreg_stg_, addr, reg_tmp1_);
    }
    vsetvli(x0, reg_vl_, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    if (dt == data_type::s8) {
        vsext_vf4(v, vreg_stg_);
        vfcvt_f_x_v(v, v);
    } else {
        vzext_vf4(v, vreg_stg_);
        vfcvt_f_xu_v(v, v);
    }
}

// Converts f32 in `v` to `dt` (with saturation/rounding for integers) and
// stores `vl` elements to `addr` (innermost stride `stride_elems`).
void jit_uni_reorder_kernel_f32_t::store_from_f32(const VReg &v,
        const Reg &addr, data_type_t dt, ptrdiff_t stride_elems) {
    const bool unit = stride_elems == 1;
    const int sz = (int)types::data_type_size(dt);

    if (dt == data_type::f32) {
        if (unit)
            vse32_v(v, addr);
        else {
            li(reg_tmp1_, (uint32_t)(stride_elems * sz));
            vsse32_v(v, addr, reg_tmp1_);
        }
        return;
    }

    if (dt == data_type::s32) {
        // INT_MAX is not representable in f32 (rounds up to 2^31). RISC-V
        // vfcvt.x.f.v saturates, so clamping to 2^31 would yield INT_MAX while
        // the reference clamps to types::max_value<float>(s32) == 2147483520.f.
        // Use the same bound to stay bit-exact with the reference.
        set_fimm(freg_tmp_, -2147483648.0f);
        vfmax_vf(v, v, freg_tmp_);
        set_fimm(freg_tmp_, types::max_value<float>(data_type::s32));
        vfmin_vf(v, v, freg_tmp_);
        vfcvt_x_f_v(v, v);
        if (unit)
            vse32_v(v, addr);
        else {
            li(reg_tmp1_, (uint32_t)(stride_elems * sz));
            vsse32_v(v, addr, reg_tmp1_);
        }
        return;
    }

    if (dt == data_type::f16) {
        vsetvli(x0, reg_vl_, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
        vfncvt_f_f_w(vreg_stg_, v); // e32/m1 -> e16/mf2 (round-to-nearest-even)
        if (unit)
            vse16_v(vreg_stg_, addr);
        else {
            li(reg_tmp1_, (uint32_t)(stride_elems * sz));
            vsse16_v(vreg_stg_, addr, reg_tmp1_);
        }
        vsetvli(x0, reg_vl_, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        return;
    }

    if (dt == data_type::bf16) {
        // f32 -> bf16 with round-to-nearest-even:
        //   u += 0x7FFF + ((u >> 16) & 1); bf16 = u >> 16
        vsrl_vi(vreg_aux_, v, 16);
        vand_vi(vreg_aux_, vreg_aux_, 1);
        li(reg_tmp1_, 0x7FFF);
        vadd_vx(v, v, reg_tmp1_);
        vadd_vv(v, v, vreg_aux_);
        vsetvli(x0, reg_vl_, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
        vnsrl_wi(vreg_stg_, v, 16);
        if (unit)
            vse16_v(vreg_stg_, addr);
        else {
            li(reg_tmp1_, (uint32_t)(stride_elems * sz));
            vsse16_v(vreg_stg_, addr, reg_tmp1_);
        }
        vsetvli(x0, reg_vl_, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        return;
    }

    // s8 / u8
    const bool is_s8 = dt == data_type::s8;
    set_fimm(freg_tmp_, is_s8 ? -128.0f : 0.0f);
    vfmax_vf(v, v, freg_tmp_);
    set_fimm(freg_tmp_, is_s8 ? 127.0f : 255.0f);
    vfmin_vf(v, v, freg_tmp_);
    if (is_s8)
        vfcvt_x_f_v(v, v);
    else
        vfcvt_xu_f_v(v, v);
    // narrow e32/m1 -> e16/mf2 -> e8/mf4 (values pre-clamped, truncation exact)
    vsetvli(x0, reg_vl_, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
    vnsrl_wi(vreg_aux_, v, 0);
    vsetvli(x0, reg_vl_, SEW::e8, LMUL::mf4, VTA::ta, VMA::ma);
    vnsrl_wi(vreg_stg_, vreg_aux_, 0);
    if (unit)
        vse8_v(vreg_stg_, addr);
    else {
        li(reg_tmp1_, (uint32_t)(stride_elems * sz));
        vsse8_v(vreg_stg_, addr, reg_tmp1_);
    }
    vsetvli(x0, reg_vl_, SEW::e32, LMUL::m1, VTA::ta,
            VMA::ma); // restore vtype for the next iter
}

// Computes the per-call running base addresses from the outer loop counters.
void jit_uni_reorder_kernel_f32_t::compute_base_addrs() {
    const bool src_many = prb_.src_scale_type == scale_type_t::MANY;
    const bool dst_many = prb_.dst_scale_type == scale_type_t::MANY;

    mv(reg_addr_, reg_ptr_in_);
    mv(reg_addr2_, reg_ptr_out_);
    if (src_many) mv(reg_saddr_src_, reg_ptr_src_scales_);
    if (dst_many) mv(reg_saddr_dst_, reg_ptr_dst_scales_);
    if (compensation_needed_) mv(reg_caddr_, reg_ptr_comp_);

    for (int d = 1; d < prb_.ndims; ++d) {
        const Reg cnt = reg_cnt_[d - 1];
        li(reg_tmp0_, (uint32_t)(prb_.nodes[d].is * itype_sz_));
        mul(reg_tmp0_, cnt, reg_tmp0_);
        add(reg_addr_, reg_addr_, reg_tmp0_);
        li(reg_tmp0_, (uint32_t)(prb_.nodes[d].os * otype_sz_));
        mul(reg_tmp0_, cnt, reg_tmp0_);
        add(reg_addr2_, reg_addr2_, reg_tmp0_);
        if (src_many || dst_many) {
            li(reg_tmp0_, (uint32_t)(prb_.nodes[d].ss * stype_sz_));
            mul(reg_tmp0_, cnt, reg_tmp0_);
            if (src_many) add(reg_saddr_src_, reg_saddr_src_, reg_tmp0_);
            if (dst_many) add(reg_saddr_dst_, reg_saddr_dst_, reg_tmp0_);
        }
        if (compensation_needed_) {
            li(reg_tmp0_,
                    (uint32_t)(prb_.nodes[d].cs * (ptrdiff_t)sizeof(int32_t)));
            mul(reg_tmp0_, cnt, reg_tmp0_);
            add(reg_caddr_, reg_caddr_, reg_tmp0_);
        }
    }
    li(reg_rem_, (uint32_t)prb_.nodes[0].n);
}

// Override reg_rem_ (= n) with the real (non-padded) element count for node[0]
// when it carries a tail, and stash it in reg_realcnt_ for the later zero-pad.
void jit_uni_reorder_kernel_f32_t::emit_node0_tail_count() {
    const auto &n0 = prb_.nodes[0];
    if (n0.is_parent_empty()) {
        // No parent => node[0] is always processed with just its tail elements.
        li(reg_rem_, (uint32_t)n0.tail_size);
    } else {
        // Parent is a driver dim (gated): the tail applies only on the parent's
        // last chunk, signalled by curr_data_chunks[parent] == 1.
        li(reg_tmp1_,
                (uint32_t)(offsetof(tail_call_param_t, curr_data_chunks)
                        + (size_t)n0.parent_node_id * sizeof(int64_t)));
        add(reg_tmp1_, reg_ptr_params_, reg_tmp1_);
        ld(reg_tmp0_, reg_tmp1_, 0);
        Label use_tail, done;
        li(reg_tmp1_, 1);
        beq(reg_tmp0_, reg_tmp1_, use_tail);
        li(reg_rem_, (uint32_t)n0.n);
        j_(done);
        L(use_tail);
        li(reg_rem_, (uint32_t)n0.tail_size);
        L(done);
    }
    if (n0.is_zero_pad_needed) mv(reg_realcnt_, reg_rem_);
}

// Zero `count` output elements (otype) contiguously starting at reg_addr2_.
void jit_uni_reorder_kernel_f32_t::emit_zero_region(const Reg &count) {
    const SEW sew = otype_sz_ == 4 ? SEW::e32
            : otype_sz_ == 2       ? SEW::e16
                                   : SEW::e8;
    Label zl, ze;
    L(zl);
    beqz(count, ze);
    vsetvli(reg_vl_, count, sew, LMUL::m1, VTA::ta, VMA::ma);
    vmv_v_i(vreg_data_, 0);
    if (otype_sz_ == 4)
        vse32_v(vreg_data_, reg_addr2_);
    else if (otype_sz_ == 2)
        vse16_v(vreg_data_, reg_addr2_);
    else
        vse8_v(vreg_data_, reg_addr2_);
    li(reg_tmp0_, (uint32_t)otype_sz_);
    mul(reg_tmp0_, reg_vl_, reg_tmp0_);
    add(reg_addr2_, reg_addr2_, reg_tmp0_);
    sub(count, count, reg_vl_);
    j_(zl);
    L(ze);
}

// After processing the real elements, zero node[0]'s (n - realcnt) padding.
// reg_addr2_ already points just past the real elements (node[0] os == 1).
void jit_uni_reorder_kernel_f32_t::emit_zero_pad() {
    li(reg_tmp0_, (uint32_t)prb_.nodes[0].n);
    sub(reg_realcnt_, reg_tmp0_, reg_realcnt_); // padcount = n - realcnt
    emit_zero_region(reg_realcnt_);
}

void jit_uni_reorder_kernel_f32_t::emit_core() {
    compute_base_addrs();
    const bool n0_tail = prb_.is_tail_present && prb_.nodes[0].tail_size > 0;
    if (n0_tail) emit_node0_tail_count();

    const auto itype = prb_.itype;
    const auto otype = prb_.otype;
    const ptrdiff_t is0 = prb_.nodes[0].is;
    const ptrdiff_t os0 = prb_.nodes[0].os;

    Label vloop, vend;
    L(vloop);
    beqz(reg_rem_, vend);
    vsetvli(reg_vl_, reg_rem_, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);

    load_to_f32(vreg_data_, reg_addr_, itype, is0);

    if (prb_.req_src_zp) vfsub_vf(vreg_data_, vreg_data_, freg_src_zp_);
    if (prb_.src_scale_type == scale_type_t::COMMON)
        vfmul_vf(vreg_data_, vreg_data_, freg_src_scale_);
    else if (prb_.src_scale_type == scale_type_t::MANY) {
        vlse32_v(vreg_scale_, reg_saddr_src_, reg_sstride_);
        vfmul_vv(vreg_data_, vreg_data_, vreg_scale_);
    }
    if (prb_.beta == 1.f) {
        load_to_f32(vreg_old_, reg_addr2_, otype, os0);
        vfadd_vv(vreg_data_, vreg_data_, vreg_old_);
    }
    if (prb_.dst_scale_type == scale_type_t::COMMON)
        vfmul_vf(vreg_data_, vreg_data_, freg_dst_scale_);
    else if (prb_.dst_scale_type == scale_type_t::MANY) {
        vlse32_v(vreg_scale_, reg_saddr_dst_, reg_sstride_);
        vfmul_vv(vreg_data_, vreg_data_, vreg_scale_);
    }
    if (prb_.req_dst_zp) vfadd_vf(vreg_data_, vreg_data_, freg_dst_zp_);
    if (prb_.scale_adjust != 1.f)
        vfmul_vf(vreg_data_, vreg_data_, freg_scale_adjust_);

    if (compensation_needed_) {
        // Accumulate the saturated integer values into the compensation buffer.
        // The driver's reduce_compensation() turns the per-thread sums into the
        // final s8s8 (-128 * sum) / asymmetric (-sum) buffers.
        float lo = 0.f, hi = 0.f;
        if (otype == data_type::u8) {
            lo = 0.f;
            hi = 255.f;
        } else if (otype == data_type::s8) {
            lo = -128.f;
            hi = 127.f;
        } else {
            lo = -2147483648.f;
            hi = types::max_value<float>(data_type::s32); // 2147483520.f
        }
        set_fimm(freg_tmp_, lo);
        vfmax_vf(vreg_data_, vreg_data_, freg_tmp_);
        set_fimm(freg_tmp_, hi);
        vfmin_vf(vreg_data_, vreg_data_, freg_tmp_);
        vfcvt_x_f_v(vreg_comp_, vreg_data_);
        if (prb_.nodes[0].cs == 0) {
            // The whole vl-group maps to one c_off: horizontal reduce + RMW.
            vmv_v_i(vreg_redzero_, 0);
            vredsum_vs(vreg_old_, vreg_comp_, vreg_redzero_);
            vmv_x_s(reg_tmp0_, vreg_old_);
            lw(reg_tmp1_, reg_caddr_, 0);
            add(reg_tmp1_, reg_tmp1_, reg_tmp0_);
            sw(reg_tmp1_, reg_caddr_, 0);
        } else {
            // Distinct c_off per lane (comp dim is innermost): strided
            // gather-add-scatter into comp[c_off + i*cs0].
            vlse32_v(vreg_old_, reg_caddr_, reg_cstride_);
            vadd_vv(vreg_old_, vreg_old_, vreg_comp_);
            vsse32_v(vreg_old_, reg_caddr_, reg_cstride_);
        }
    }

    store_from_f32(vreg_data_, reg_addr2_, otype, os0);

    mul(reg_tmp0_, reg_vl_, reg_istride_);
    add(reg_addr_, reg_addr_, reg_tmp0_);
    mul(reg_tmp0_, reg_vl_, reg_ostride_);
    add(reg_addr2_, reg_addr2_, reg_tmp0_);
    if (prb_.src_scale_type == scale_type_t::MANY) {
        mul(reg_tmp0_, reg_vl_, reg_sstride_);
        add(reg_saddr_src_, reg_saddr_src_, reg_tmp0_);
    }
    if (prb_.dst_scale_type == scale_type_t::MANY) {
        mul(reg_tmp0_, reg_vl_, reg_sstride_);
        add(reg_saddr_dst_, reg_saddr_dst_, reg_tmp0_);
    }
    if (compensation_needed_ && prb_.nodes[0].cs != 0) {
        mul(reg_tmp0_, reg_vl_, reg_cstride_);
        add(reg_caddr_, reg_caddr_, reg_tmp0_);
    }
    sub(reg_rem_, reg_rem_, reg_vl_);
    j_(vloop);
    L(vend);

    if (n0_tail && prb_.nodes[0].is_zero_pad_needed) emit_zero_pad();
}

void jit_uni_reorder_kernel_f32_t::emit_pure_copy_core() {
    compute_base_addrs();
    const bool n0_tail = prb_.is_tail_present && prb_.nodes[0].tail_size > 0;
    if (n0_tail) emit_node0_tail_count();

    const ptrdiff_t is0 = prb_.nodes[0].is;
    const ptrdiff_t os0 = prb_.nodes[0].os;
    const bool in_unit = is0 == 1;
    const bool out_unit = os0 == 1;
    const SEW sew = itype_sz_ == 4 ? SEW::e32
            : itype_sz_ == 2       ? SEW::e16
                                   : SEW::e8;

    Label vloop, vend;
    L(vloop);
    beqz(reg_rem_, vend);
    vsetvli(reg_vl_, reg_rem_, sew, LMUL::m1, VTA::ta, VMA::ma);

    if (sew == SEW::e32) {
        if (in_unit)
            vle32_v(vreg_data_, reg_addr_);
        else
            vlse32_v(vreg_data_, reg_addr_, reg_istride_);
        if (out_unit)
            vse32_v(vreg_data_, reg_addr2_);
        else
            vsse32_v(vreg_data_, reg_addr2_, reg_ostride_);
    } else if (sew == SEW::e16) {
        if (in_unit)
            vle16_v(vreg_data_, reg_addr_);
        else
            vlse16_v(vreg_data_, reg_addr_, reg_istride_);
        if (out_unit)
            vse16_v(vreg_data_, reg_addr2_);
        else
            vsse16_v(vreg_data_, reg_addr2_, reg_ostride_);
    } else {
        if (in_unit)
            vle8_v(vreg_data_, reg_addr_);
        else
            vlse8_v(vreg_data_, reg_addr_, reg_istride_);
        if (out_unit)
            vse8_v(vreg_data_, reg_addr2_);
        else
            vsse8_v(vreg_data_, reg_addr2_, reg_ostride_);
    }

    mul(reg_tmp0_, reg_vl_, reg_istride_);
    add(reg_addr_, reg_addr_, reg_tmp0_);
    mul(reg_tmp0_, reg_vl_, reg_ostride_);
    add(reg_addr2_, reg_addr2_, reg_tmp0_);
    sub(reg_rem_, reg_rem_, reg_vl_);
    j_(vloop);
    L(vend);

    if (n0_tail && prb_.nodes[0].is_zero_pad_needed) emit_zero_pad();
}

void jit_uni_reorder_kernel_f32_t::emit_reorder_loops(int level) {
    if (level <= 0) {
        if (interim_f32_)
            emit_core();
        else
            emit_pure_copy_core();
        return;
    }

    const Reg cnt = reg_cnt_[level - 1];
    li(cnt, 0);
    Label lp;
    L(lp);
    emit_reorder_loops(level - 1);
    addi(cnt, cnt, 1);
    li(reg_tmp0_, (uint32_t)prb_.nodes[level].n);
    blt(cnt, reg_tmp0_, lp);
}

void jit_uni_reorder_kernel_f32_t::generate() {
    preamble();

    mv(reg_ptr_params_, a0);
    ld(reg_ptr_in_, reg_ptr_params_, (int)offsetof(call_param_t, in));
    ld(reg_ptr_out_, reg_ptr_params_, (int)offsetof(call_param_t, out));

    if (prb_.src_scale_type != scale_type_t::NONE) {
        ld(reg_ptr_src_scales_, reg_ptr_params_,
                (int)offsetof(call_param_t, src_scales));
        if (prb_.src_scale_type == scale_type_t::COMMON)
            flw(freg_src_scale_, reg_ptr_src_scales_, 0);
    }
    if (prb_.dst_scale_type != scale_type_t::NONE) {
        ld(reg_ptr_dst_scales_, reg_ptr_params_,
                (int)offsetof(call_param_t, dst_scales));
        if (prb_.dst_scale_type == scale_type_t::COMMON)
            flw(freg_dst_scale_, reg_ptr_dst_scales_, 0);
    }
    if (prb_.req_src_zp) {
        lw(reg_tmp0_, reg_ptr_params_, (int)offsetof(call_param_t, src_zp));
        fcvt_s_w(freg_src_zp_, reg_tmp0_);
    }
    if (prb_.req_dst_zp) {
        lw(reg_tmp0_, reg_ptr_params_, (int)offsetof(call_param_t, dst_zp));
        fcvt_s_w(freg_dst_zp_, reg_tmp0_);
    }
    if (compensation_needed_)
        ld(reg_ptr_comp_, reg_ptr_params_,
                (int)offsetof(call_param_t, compensation_scratch));
    if (prb_.scale_adjust != 1.f)
        set_fimm(freg_scale_adjust_, prb_.scale_adjust);

    // innermost (node[0]) byte strides, used for vector (de)interleaving and
    // for advancing the running addresses inside the vl-loop.
    li(reg_istride_, (uint32_t)(prb_.nodes[0].is * itype_sz_));
    li(reg_ostride_, (uint32_t)(prb_.nodes[0].os * otype_sz_));
    if (prb_.src_scale_type == scale_type_t::MANY
            || prb_.dst_scale_type == scale_type_t::MANY)
        li(reg_sstride_, (uint32_t)(prb_.nodes[0].ss * stype_sz_));
    if (compensation_needed_ && prb_.nodes[0].cs != 0)
        li(reg_cstride_,
                (uint32_t)(prb_.nodes[0].cs * (ptrdiff_t)sizeof(int32_t)));

    emit_reorder_loops(prb_.ndims - 1);

    postamble();
}

} // namespace tr
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
