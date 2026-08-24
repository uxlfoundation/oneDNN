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
#include "common/type_helpers.hpp"

#include "cpu/rv64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace binary_injector {

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    switch (alg) {
        case binary_add:
        case binary_sub:
        case binary_mul:
        case binary_div:
        case binary_max:
        case binary_min:
        case binary_ge:
        case binary_gt:
        case binary_le:
        case binary_lt:
        case binary_eq:
        case binary_ne: return true;
        default: return false;
    }
}

} // namespace binary_injector

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx, const binary_injector::rhs_arg_dynamic_params_t &dyn) {
    for (size_t i = start_idx; i < end_idx; i++)
        compute_vector(i, dyn);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector(
        size_t idx, const binary_injector::rhs_arg_dynamic_params_t &dyn) {
    // scalar broadcast ignores the offset; other strategies need the per-vmm
    // output element offset to compute the rhs address.
    Reg out_off = x0;
    const auto it = dyn.vmm_idx_to_out_off.find((int)idx);
    if (it != dyn.vmm_idx_to_out_off.end())
        out_off = it->second;
    else
        assert(rhs_.bcast == binary_injector::broadcast_t::scalar
                && IMPLICATION(select_ != nullptr,
                        select_->bcast == binary_injector::broadcast_t::scalar)
                && "non-scalar dynamic binary rhs needs an out-offset");
    if (alg_ == alg_kind::binary_select)
        apply_select(Vmm(idx), out_off);
    else {
        const bool scalar = load_operand(out_off, rhs_);
        apply_op(Vmm(idx), scalar);
    }
}

// Load the rhs slice into f_rhs_ (scalar broadcast) or v_rhs_ (vector) from
// base + strategy(out_off), converting rhs_dt_ to f32. gpr_ and off_ are address
// scratch.
// Returns whether the result is a broadcast scalar (true) vs a vector (false).
//
// Non-f32 rhs and the gather are only enabled by the binary primitive, which
// computes at e32/m4; the s8/u8/xf16 paths briefly switch SEW (e8/m1, e16/m2 and
// e32/m4 share VLMAX, so `vsetvli x0,x0` keeps vl) and widen v_rhs_ in place,
// restoring e32/m4. v_idx_ holds the gather byte-index; other consumers use only
// scalar/per_element f32, which take the no-switch paths.
template <cpu_isa_t isa>
bool jit_uni_binary_injector_t<isa>::load_operand(
        const Reg &out_off, operand_t &op) {
    using bt = binary_injector::broadcast_t;
    // Loads the per-binary base pointer (indirect) into gpr_, or returns the
    // direct pointer. Call after any use of gpr_ for channel math.
    auto load_base = [&]() -> Reg {
        if (op.indirect) {
            h_->ld(op.gpr, op.rhs_addr, op.arg_idx * (int)sizeof(void *));
            return op.gpr;
        }
        return op.rhs_addr;
    };
    const int esz = (int)types::data_type_size(op.rhs_dt);
    const int sh = esz == 4 ? 2 : (esz == 2 ? 1 : 0);

    // Load vl elements of rhs_dt_ into v_rhs_ as f32 (e32/m4). is_gather uses the
    // byte-index vector v_idx_ (vluxei32); else a contiguous / strided load from
    // `base`. Narrow dtypes switch SEW (vl preserved) and widen v_rhs_ in place.
    auto load_rhs_vec = [&](const Reg &base, bool is_gather) {
        if (op.rhs_dt != data_type::f32 && op.rhs_dt != data_type::s32) {
            assert(op.v_tmp != VReg(0)
                    && "narrow binary rhs requires explicit staging scratch");
            assert(op.v_tmp != op.v_rhs
                    && "binary rhs staging scratch overlaps rhs result");
            assert(IMPLICATION(is_gather, op.v_tmp != op.v_idx)
                    && "binary rhs staging scratch overlaps gather index");
        }
        if (op.rhs_dt == data_type::f32 || op.rhs_dt == data_type::s32) {
            if (is_gather)
                h_->vluxei32_v(op.v_rhs, base, op.v_idx);
            else if (op.strided)
                h_->vlse32_v(op.v_rhs, base, op.rhs_stride);
            else
                h_->vle32_v(op.v_rhs, base);
            if (op.rhs_dt == data_type::s32)
                h_->vfcvt_f_x_v(op.v_rhs, op.v_rhs);
        } else if (op.rhs_dt == data_type::s8 || op.rhs_dt == data_type::u8) {
            // Load bytes into v_tmp_, then widen into v_rhs_ (a widening op may
            // not overlap its source's low part, so stage in a distinct group).
            h_->vsetvli(x0, x0, SEW::e8, LMUL::m1, VTA::ta, VMA::ma); // keep vl
            if (is_gather)
                h_->vluxei32_v(op.v_tmp, base, op.v_idx);
            else if (op.strided)
                h_->vlse8_v(op.v_tmp, base, op.rhs_stride);
            else
                h_->vle8_v(op.v_tmp, base);
            h_->vsetvli(x0, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
            if (op.rhs_dt == data_type::s8) {
                h_->vsext_vf4(op.v_rhs, op.v_tmp);
                h_->vfcvt_f_x_v(op.v_rhs, op.v_rhs);
            } else {
                h_->vzext_vf4(op.v_rhs, op.v_tmp);
                h_->vfcvt_f_xu_v(op.v_rhs, op.v_rhs);
            }
        } else { // f16 / bf16
            h_->vsetvli(
                    x0, x0, SEW::e16, LMUL::m2, VTA::ta, VMA::ma); // keep vl
            if (is_gather)
                h_->vluxei32_v(op.v_tmp, base, op.v_idx);
            else if (op.strided)
                h_->vlse16_v(op.v_tmp, base, op.rhs_stride);
            else
                h_->vle16_v(op.v_tmp, base);
            // e16m2 -> e32m4; Zvfbfmin for bf16, Zvfh for f16.
            if (op.rhs_dt == data_type::bf16)
                h_->vfwcvtbf16_f_f_v(op.v_rhs, op.v_tmp);
            else
                h_->vfwcvt_f_f_v(op.v_rhs, op.v_tmp);
            h_->vsetvli(x0, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        }
    };

    if (op.bcast == bt::scalar) {
        // One rhs value broadcast to all lanes. Integer/f32 use scalar loads ->
        // f_rhs_ (no vl change); f16/bf16 broadcast-load (stride 0) -> v_rhs_.
        const Reg base = load_base();
        if (op.rhs_dt == data_type::f32) {
            h_->flw(op.f_rhs, base, 0);
            return true;
        }
        if (op.rhs_dt == data_type::s32) {
            h_->lw(op.gpr, base, 0);
            h_->fcvt_s_w(op.f_rhs, op.gpr);
            return true;
        }
        if (op.rhs_dt == data_type::s8) {
            h_->lb(op.gpr, base, 0);
            h_->fcvt_s_w(op.f_rhs, op.gpr);
            return true;
        }
        if (op.rhs_dt == data_type::u8) {
            h_->lbu(op.gpr, base, 0);
            h_->fcvt_s_wu(op.f_rhs, op.gpr);
            return true;
        }
        // f16/bf16: broadcast one value to every lane (stride 0) into v_tmp_,
        // then widen into v_rhs_ (staged, a widen cannot overlap its source).
        assert(op.v_tmp != VReg(0)
                && "xf16 binary rhs requires explicit staging scratch");
        assert(op.v_tmp != op.v_rhs
                && "binary rhs staging scratch overlaps rhs result");
        h_->vsetvli(x0, x0, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
        h_->vlse16_v(op.v_tmp, base, x0);
        if (op.rhs_dt == data_type::bf16)
            h_->vfwcvtbf16_f_f_v(op.v_rhs, op.v_tmp);
        else
            h_->vfwcvt_f_f_v(op.v_rhs, op.v_tmp);
        h_->vsetvli(x0, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        return false;
    }
    if (op.bcast == bt::per_oc) {
        assert(op.v_idx != VReg(0)
                && "per-oc binary rhs requires explicit index scratch");
        assert(op.v_idx != op.v_rhs
                && "binary rhs gather index overlaps rhs result");
        // Per-lane gather index (element): a vl-run may span dim boundaries, so
        // each lane resolves its own broadcast-dim index.
        //   idx = ((o / oc_stride) % C) * blk + (o % blk), o = out_off + lane
        // Covers per_oc / per_w (blk=1) and blocked per_oc (blk = inner block).
        h_->vid_v(op.v_idx);
        h_->vadd_vx(op.v_idx, op.v_idx, out_off); // o
        if (op.blk > 1)
            h_->vand_vi(op.v_rhs, op.v_idx, (int)(op.blk - 1)); // low=o%blk
        if (op.oc_stride != 1) {
            h_->li(op.gpr, op.oc_stride);
            h_->vdivu_vx(op.v_idx, op.v_idx, op.gpr);
        }
        h_->li(op.gpr, op.C);
        h_->vremu_vx(op.v_idx, op.v_idx, op.gpr); // outer index
        if (op.blk > 1) {
            int lg = 0;
            for (dim_t b = op.blk; b > 1; b >>= 1)
                lg++;
            h_->vsll_vi(op.v_idx, op.v_idx, lg); // outer * blk
            h_->vadd_vv(op.v_idx, op.v_idx, op.v_rhs); // + low
        }
        if (sh) h_->vsll_vi(op.v_idx, op.v_idx, sh); // * esz -> byte offset
        load_rhs_vec(load_base(), /*is_gather=*/true);
        return false;
    }
    // per_element: address of the first active lane. Byte-offset consumers
    // (indirect, f32) add the offset in place; element-offset consumers scale by
    // sizeof(rhs_dt_) into off_.
    Reg addr;
    if (op.off_is_bytes) {
        h_->ld(op.gpr, op.rhs_addr, op.arg_idx * (int)sizeof(void *));
        h_->add(op.gpr, op.gpr, out_off);
        addr = op.gpr;
    } else {
        h_->slli(op.off, out_off, sh);
        h_->add(op.off, load_base(), op.off);
        addr = op.off;
    }
    load_rhs_vec(addr, /*is_gather=*/false);
    return false;
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::materialize_cmp(const Vmm &dst) {
    // v0 contains the comparison result. Materialize the public binary
    // contract (0.0f/1.0f), reusing the injector's scalar/GPR scratch only
    // after the rhs comparison has consumed them.
    h_->fmv_w_x(rhs_.f_rhs, x0);
    h_->vfmv_v_f(dst, rhs_.f_rhs);
    h_->li(rhs_.gpr, 0x3f800000);
    h_->fmv_w_x(rhs_.f_rhs, rhs_.gpr);
    h_->vfmerge_vfm(dst, dst, rhs_.f_rhs);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::apply_select(
        const Vmm &dst, const Reg &out_off) {
    assert(select_ != nullptr);

    // Preserve src1 in its vector scratch while loading the independent select
    // condition. The two operands may use different dtypes and broadcasts.
    const bool rhs_scalar = load_operand(out_off, rhs_);
    if (rhs_scalar) h_->vfmv_v_f(rhs_.v_rhs, rhs_.f_rhs);

    operand_t &cond = *select_;
    const bool cond_scalar = load_operand(out_off, cond);
    if (cond_scalar) h_->vfmv_v_f(cond.v_rhs, cond.f_rhs);

    // oneDNN select is condition ? accumulator : src1. Numeric zero (including
    // -0) is false; every nonzero value, including NaN, is true.
    h_->fmv_w_x(cond.f_rhs, x0);
    h_->vmfne_vf(VReg(0), cond.v_rhs, cond.f_rhs);
    h_->vmerge_vvm(dst, rhs_.v_rhs, dst);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::apply_op(const Vmm &dst, bool scalar) {
    using namespace alg_kind;
    // dst = dst OP rhs (src0 is the accumulator, src1 the rhs).
    switch (alg_) {
        case binary_add:
            if (scalar)
                h_->vfadd_vf(dst, dst, rhs_.f_rhs);
            else
                h_->vfadd_vv(dst, dst, rhs_.v_rhs);
            break;
        case binary_sub:
            if (scalar)
                h_->vfsub_vf(dst, dst, rhs_.f_rhs);
            else
                h_->vfsub_vv(dst, dst, rhs_.v_rhs);
            break;
        case binary_mul:
            if (scalar)
                h_->vfmul_vf(dst, dst, rhs_.f_rhs);
            else
                h_->vfmul_vv(dst, dst, rhs_.v_rhs);
            break;
        case binary_div:
            if (scalar)
                h_->vfdiv_vf(dst, dst, rhs_.f_rhs);
            else
                h_->vfdiv_vv(dst, dst, rhs_.v_rhs);
            break;
        case binary_max:
            // nstl::max(dst,rhs) = (rhs < dst) ? dst : rhs (picks rhs on
            // ties/unordered, matching the reference and x86 vmaxps).
            if (scalar) h_->vfmv_v_f(rhs_.v_rhs, rhs_.f_rhs);
            h_->vmflt_vv(VReg(0), rhs_.v_rhs, dst);
            h_->vmerge_vvm(dst, rhs_.v_rhs, dst);
            break;
        case binary_min:
            // nstl::min(dst,rhs) = (dst < rhs) ? dst : rhs.
            if (scalar) h_->vfmv_v_f(rhs_.v_rhs, rhs_.f_rhs);
            h_->vmflt_vv(VReg(0), dst, rhs_.v_rhs);
            h_->vmerge_vvm(dst, rhs_.v_rhs, dst);
            break;
        case binary_ge:
            if (scalar)
                h_->vmfge_vf(VReg(0), dst, rhs_.f_rhs);
            else
                h_->vmfge_vv(VReg(0), dst, rhs_.v_rhs);
            materialize_cmp(dst);
            break;
        case binary_gt:
            if (scalar)
                h_->vmfgt_vf(VReg(0), dst, rhs_.f_rhs);
            else
                h_->vmfgt_vv(VReg(0), dst, rhs_.v_rhs);
            materialize_cmp(dst);
            break;
        case binary_le:
            if (scalar)
                h_->vmfle_vf(VReg(0), dst, rhs_.f_rhs);
            else
                h_->vmfle_vv(VReg(0), dst, rhs_.v_rhs);
            materialize_cmp(dst);
            break;
        case binary_lt:
            if (scalar)
                h_->vmflt_vf(VReg(0), dst, rhs_.f_rhs);
            else
                h_->vmflt_vv(VReg(0), dst, rhs_.v_rhs);
            materialize_cmp(dst);
            break;
        case binary_eq:
            if (scalar)
                h_->vmfeq_vf(VReg(0), dst, rhs_.f_rhs);
            else
                h_->vmfeq_vv(VReg(0), dst, rhs_.v_rhs);
            materialize_cmp(dst);
            break;
        case binary_ne:
            if (scalar)
                h_->vmfne_vf(VReg(0), dst, rhs_.f_rhs);
            else
                h_->vmfne_vv(VReg(0), dst, rhs_.v_rhs);
            materialize_cmp(dst);
            break;
        default: assert(!"unsupported binary alg"); break;
    }
}

template struct jit_uni_binary_injector_t<v>;
template struct jit_uni_binary_injector_t<zvfh>;
template struct jit_uni_binary_injector_t<zvfbfwma>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
