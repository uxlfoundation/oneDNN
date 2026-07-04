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
#include <cstddef>
#include <cstring>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_uni_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

// Set the vtype for loading `dt` (narrow element width); returns nothing, vl in
// t0. Loads `vl` from len in a4.
void set_load_vtype(
        jit_generator_t *h, data_type_t dt, const Reg &vl, const Reg &len) {
    if (dt == data_type::f32 || dt == data_type::s32)
        h->vsetvli(vl, len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    else if (dt == data_type::f16)
        h->vsetvli(vl, len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    else
        h->vsetvli(vl, len, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
}

// Switch to the f32 compute vtype for `dt` (m1 for f32/s32, m2 for f16, m4 for
// s8/u8), keeping vl.
void set_compute_vtype(jit_generator_t *h, data_type_t dt, const Reg &vl) {
    if (dt == data_type::f32 || dt == data_type::s32)
        h->vsetvli(x0, vl, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    else if (dt == data_type::f16)
        h->vsetvli(x0, vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
    else
        h->vsetvli(x0, vl, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
}

// Load one chunk from `ptr` (narrow vtype already set) into `vdst` and widen to
// f32 (compute vtype must be set afterwards by the caller for f16/int).
// `stage` is an 8/16-bit staging group. For f32 there is no conversion.
void load_widen(jit_generator_t *h, data_type_t dt, const VReg &vdst,
        const VReg &stage, const Reg &ptr) {
    if (dt == data_type::f32) {
        h->vle32_v(vdst, ptr);
    } else if (dt == data_type::s32) {
        h->vle32_v(vdst, ptr); // cvt done after compute-vtype switch
    } else if (dt == data_type::f16) {
        h->vle16_v(stage, ptr);
        h->vfwcvt_f_f_v(vdst, stage); // e16m1 -> e32m2 (at e16m1 vtype)
    } else {
        h->vle8_v(stage, ptr); // widened after compute-vtype switch
    }
}

void emit_binary_loop(jit_generator_t *h, alg_kind_t alg, bool scalar_src1,
        const post_ops_t &post_ops, data_type_t dt) {
    using namespace alg_kind;
    const Reg p0 = a1, p1 = a2, pd = a3, len = a4, vl = t0, bytes = t1,
              gpr = t2;
    const VReg vd(4), vs1(20), vt(2), vt2(3);
    const FReg f_s1 = fa3, fc = fa4;

    eltwise_injector::static_params_t esp(
            VReg(8), VReg(12), VReg(16), fa0, fa1, gpr, /*is_fwd=*/true);
    injector::jit_uni_postops_injector_t<v> po_inj(h, post_ops, esp);

    const bool is_s8 = dt == data_type::s8;
    const bool is_u8 = dt == data_type::u8;
    const bool is_i8 = is_s8 || is_u8;
    const bool is_int = dt == data_type::s32 || is_i8;

    auto setf = [&](float val) {
        uint32_t b;
        std::memcpy(&b, &val, sizeof(b));
        h->li(gpr, b);
        h->fmv_w_x(fc, gpr);
    };

    // Pre-load the broadcast scalar src1 into f_s1 (as f32), once.
    if (scalar_src1) {
        if (dt == data_type::f32) {
            h->flw(f_s1, p1, 0);
        } else if (dt == data_type::s32) {
            h->lw(gpr, p1, 0);
            h->fcvt_s_w(f_s1, gpr);
        } else if (dt == data_type::s8) {
            h->lb(gpr, p1, 0);
            h->fcvt_s_w(f_s1, gpr);
        } else if (dt == data_type::u8) {
            h->lbu(gpr, p1, 0);
            h->fcvt_s_wu(f_s1, gpr);
        } else { // f16: widen via the vector unit, extract lane 0
            h->li(gpr, 1);
            h->vsetvli(x0, gpr, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
            h->vle16_v(vt, p1);
            h->vfwcvt_f_f_v(vs1, vt); // e16m1 -> e32m2
            // vfmv_f_s reads lane 0 at the CURRENT SEW: switch to e32 first, or
            // it would extract 16 bits of the widened f32 (wrong scalar).
            h->vsetvli(x0, gpr, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
            h->vfmv_f_s(f_s1, vs1);
        }
    }

    // Turn the comparison mask in v0 into a 0.0/1.0 result in vd (in the active
    // f32 compute vtype). The data/aux groups stay away from v0 so later
    // post-op injectors can also use it as a mask; fa5/fa6 are free here.
    auto cmp_to_01 = [&]() {
        const FReg f_zero = fa5, f_one = fa6;
        h->fmv_w_x(f_zero, x0); // 0.0
        h->li(gpr, 0x3f800000); // 1.0f bit pattern
        h->fmv_w_x(f_one, gpr);
        h->vfmv_v_f(vd, f_zero);
        h->vfmerge_vfm(vd, vd, f_one); // vd[i] = v0[i] ? 1.0 : 0.0
    };

    auto binop_vf = [&](const FReg &f) {
        switch (alg) {
            case binary_add: h->vfadd_vf(vd, vd, f); break;
            case binary_sub: h->vfsub_vf(vd, vd, f); break;
            case binary_mul: h->vfmul_vf(vd, vd, f); break;
            case binary_div: h->vfdiv_vf(vd, vd, f); break;
            case binary_max:
                h->vfmv_v_f(vs1, f);
                h->vmflt_vv(VReg(0), vd, vs1);
                h->vmerge_vvm(vd, vd, vs1);
                h->vmfne_vv(VReg(0), vs1, vs1);
                h->vmerge_vvm(vd, vd, vs1);
                break;
            case binary_min:
                h->vfmv_v_f(vs1, f);
                h->vmflt_vv(VReg(0), vs1, vd);
                h->vmerge_vvm(vd, vd, vs1);
                h->vmfne_vv(VReg(0), vs1, vs1);
                h->vmerge_vvm(vd, vd, vs1);
                break;
            // Comparison: dst = (dst OP src1) ? 1 : 0 (vf form has gt/ge).
            case binary_ge:
                h->vmfge_vf(VReg(0), vd, f);
                cmp_to_01();
                break;
            case binary_gt:
                h->vmfgt_vf(VReg(0), vd, f);
                cmp_to_01();
                break;
            case binary_le:
                h->vmfle_vf(VReg(0), vd, f);
                cmp_to_01();
                break;
            case binary_lt:
                h->vmflt_vf(VReg(0), vd, f);
                cmp_to_01();
                break;
            case binary_eq:
                h->vmfeq_vf(VReg(0), vd, f);
                cmp_to_01();
                break;
            case binary_ne:
                h->vmfne_vf(VReg(0), vd, f);
                cmp_to_01();
                break;
            default: break;
        }
    };
    auto binop_vv = [&](const VReg &v1) {
        switch (alg) {
            case binary_add: h->vfadd_vv(vd, vd, v1); break;
            case binary_sub: h->vfsub_vv(vd, vd, v1); break;
            case binary_mul: h->vfmul_vv(vd, vd, v1); break;
            case binary_div: h->vfdiv_vv(vd, vd, v1); break;
            case binary_max:
                h->vmflt_vv(VReg(0), vd, v1);
                h->vmerge_vvm(vd, vd, v1);
                h->vmfne_vv(VReg(0), v1, v1);
                h->vmerge_vvm(vd, vd, v1);
                break;
            case binary_min:
                h->vmflt_vv(VReg(0), v1, vd);
                h->vmerge_vvm(vd, vd, v1);
                h->vmfne_vv(VReg(0), v1, v1);
                h->vmerge_vvm(vd, vd, v1);
                break;
            // Comparison: vmfgt/vmfge have no vv form, so swap operands
            // (dst > v1 == v1 < dst, dst >= v1 == v1 <= dst).
            case binary_ge:
                h->vmfle_vv(VReg(0), v1, vd);
                cmp_to_01();
                break;
            case binary_gt:
                h->vmflt_vv(VReg(0), v1, vd);
                cmp_to_01();
                break;
            case binary_le:
                h->vmfle_vv(VReg(0), vd, v1);
                cmp_to_01();
                break;
            case binary_lt:
                h->vmflt_vv(VReg(0), vd, v1);
                cmp_to_01();
                break;
            case binary_eq:
                h->vmfeq_vv(VReg(0), vd, v1);
                cmp_to_01();
                break;
            case binary_ne:
                h->vmfne_vv(VReg(0), vd, v1);
                cmp_to_01();
                break;
            default: break;
        }
    };

    // Ternary select: dst = src2 ? src0 : src1, done in the native dtype (no
    // f32 round-trip, so no precision loss). src2 is an s8 boolean mask.
    if (alg == binary_select) {
        const Reg p2 = a5;
        const SEW nsew = (dt == data_type::f32 || dt == data_type::s32)
                ? SEW::e32
                : (dt == data_type::f16 ? SEW::e16 : SEW::e8);
        auto vld = [&](const VReg &vv, const Reg &p) {
            if (nsew == SEW::e32)
                h->vle32_v(vv, p);
            else if (nsew == SEW::e16)
                h->vle16_v(vv, p);
            else
                h->vle8_v(vv, p);
        };
        Label sloop, sdone;
        h->L(sloop);
        h->beqz(len, sdone);
        h->vsetvli(vl, len, nsew, LMUL::m1, VTA::ta, VMA::ma);
        vld(vd, p0);
        vld(vs1, p1);
        h->vsetvli(x0, vl, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
        h->vle8_v(vt, p2);
        h->vmsne_vx(VReg(0), vt, x0); // v0 = (src2 != 0)
        h->vsetvli(x0, vl, nsew, LMUL::m1, VTA::ta, VMA::ma);
        h->vmerge_vvm(vt2, vs1, vd); // v0 ? src0(vd) : src1(vs1)
        if (nsew == SEW::e32)
            h->vse32_v(vt2, pd);
        else if (nsew == SEW::e16)
            h->vse16_v(vt2, pd);
        else
            h->vse8_v(vt2, pd);
        const int dsz = static_cast<int>(types::data_type_size(dt));
        const int sh = dsz == 4 ? 2 : (dsz == 2 ? 1 : 0);
        if (sh)
            h->slli(bytes, vl, sh);
        else
            h->mv(bytes, vl);
        h->add(p0, p0, bytes);
        h->add(p1, p1, bytes);
        h->add(p2, p2, vl); // src2 is s8: one byte per element
        h->add(pd, pd, bytes);
        h->sub(len, len, vl);
        h->j_(sloop);
        h->L(sdone);
        return;
    }

    Label loop, done;
    h->L(loop);
    h->beqz(len, done);

    // ---- load src0 (+ per-element src1) at the narrow vtype, then widen ----
    set_load_vtype(h, dt, vl, len);
    load_widen(h, dt, vd, vt, p0);
    if (!scalar_src1) load_widen(h, dt, vs1, vt2, p1);

    if (is_int) {
        set_compute_vtype(h, dt, vl);
        if (is_i8) {
            if (is_s8) {
                h->vsext_vf4(vd, vt);
                if (!scalar_src1) h->vsext_vf4(vs1, vt2);
            } else {
                h->vzext_vf4(vd, vt);
                if (!scalar_src1) h->vzext_vf4(vs1, vt2);
            }
        }
        if (is_u8) {
            h->vfcvt_f_xu_v(vd, vd);
            if (!scalar_src1) h->vfcvt_f_xu_v(vs1, vs1);
        } else { // s32 / s8
            h->vfcvt_f_x_v(vd, vd);
            if (!scalar_src1) h->vfcvt_f_x_v(vs1, vs1);
        }
    } else if (dt == data_type::f16) {
        set_compute_vtype(h, dt, vl); // e32m2
    }

    // ---- binary op + eltwise post-ops ----
    if (scalar_src1)
        binop_vf(f_s1);
    else
        binop_vv(vs1);
    po_inj.compute_vector(vd.getIdx());

    // ---- convert back + store (compute vtype active) ----
    if (dt == data_type::f32) {
        h->vse32_v(vd, pd);
    } else if (dt == data_type::s32) {
        setf(-2147483648.0f);
        h->vfmax_vf(vd, vd, fc);
        setf(2147483647.0f);
        h->vfmin_vf(vd, vd, fc);
        h->vfcvt_x_f_v(vd, vd);
        h->vse32_v(vd, pd);
    } else if (dt == data_type::f16) {
        h->vsetvli(x0, vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        h->vfncvt_f_f_w(vt, vd);
        h->vse16_v(vt, pd);
    } else { // s8 / u8
        setf(is_s8 ? -128.0f : 0.0f);
        h->vfmax_vf(vd, vd, fc);
        setf(is_s8 ? 127.0f : 255.0f);
        h->vfmin_vf(vd, vd, fc);
        if (is_s8)
            h->vfcvt_x_f_v(vd, vd);
        else
            h->vfcvt_xu_f_v(vd, vd);
        h->vsetvli(x0, vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
        h->vnsrl_wi(vs1, vd, 0); // i32m4 -> i16m2 (vs1 reused as temp)
        h->vsetvli(x0, vl, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
        h->vnsrl_wi(vt, vs1, 0); // i16m2 -> i8m1
        h->vse8_v(vt, pd);
    }

    // ---- advance and loop ----
    const int dsz = static_cast<int>(types::data_type_size(dt));
    const int sh = dsz == 4 ? 2 : (dsz == 2 ? 1 : 0);
    if (sh)
        h->slli(bytes, vl, sh);
    else
        h->mv(bytes, vl);
    h->add(p0, p0, bytes);
    h->add(pd, pd, bytes);
    if (!scalar_src1) h->add(p1, p1, bytes);
    h->sub(len, len, vl);
    h->j_(loop);
    h->L(done);
    MAYBE_UNUSED(is_int);
}

} // namespace

struct jit_uni_binary_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src0;
        const void *src1;
        const void *src2; // ternary select mask (s8), else null
        void *dst;
        dim_t len;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binary_kernel_t)

    jit_uni_binary_kernel_t(alg_kind_t alg, bool scalar_src1,
            const post_ops_t &post_ops, data_type_t dt)
        : jit_generator_t("jit_uni_binary")
        , alg_(alg)
        , scalar_src1_(scalar_src1)
        , post_ops_(post_ops)
        , dt_(dt) {
        create_kernel();
    }
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }
    void generate() override {
        ld(a1, a0, 0); // src0
        ld(a2, a0, 8); // src1
        ld(a5, a0, 16); // src2 (select mask)
        ld(a3, a0, 24); // dst
        ld(a4, a0, 32); // len
        emit_binary_loop(this, alg_, scalar_src1_, post_ops_, dt_);
        ret();
    }
    alg_kind_t alg_;
    bool scalar_src1_;
    post_ops_t post_ops_;
    data_type_t dt_;
};

jit_uni_binary_t::jit_uni_binary_t(const pd_t *apd) : primitive_t(apd) {}
jit_uni_binary_t::~jit_uni_binary_t() = default;

status_t jit_uni_binary_t::init(engine_t *engine) {
    UNUSED(engine);
    const bool scalar = pd()->bcast_ == bcast_t::scalar
            || pd()->bcast_ == bcast_t::per_oc_blocked;
    kernel_.reset(new jit_uni_binary_kernel_t(pd()->desc()->alg_kind, scalar,
            pd()->attr()->post_ops_, pd()->dst_md()->data_type));
    return status::success;
}

status_t jit_uni_binary_t::execute(const exec_ctx_t &ctx) const {
    const auto *s0 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_0);
    const auto *s1 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_1);
    auto *d = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const data_type_t dt = pd()->dst_md()->data_type;
    const size_t es = types::data_type_size(dt);
    const dim_t block = pd()->block_len_;
    const dim_t nb = pd()->n_blocks_;
    const bcast_t bc = pd()->bcast_;
    const dim_t C = pd()->dst_md()->ndims >= 2 ? pd()->dst_md()->dims[1] : 1;
    // ternary select mask (s8, 1 byte/elem, no broadcast -> flat path)
    const bool is_select = pd()->desc()->alg_kind == alg_kind::binary_select;
    const char *s2
            = is_select ? CTX_IN_MEM(const char *, DNNL_ARG_SRC_2) : nullptr;

    // Honor non-zero md offsets (dense submemory views): execute() indexes flat
    // from each base, so shift every operand by its own offset0.
    s0 += (size_t)pd()->src_md(0)->offset0 * es;
    s1 += (size_t)pd()->src_md(1)->offset0 * es;
    d += (size_t)pd()->dst_md()->offset0 * es;
    if (s2) s2 += (size_t)pd()->src_md(2)->offset0; // s8 mask, 1 byte/elem

    if (bc == bcast_t::scalar || bc == bcast_t::none) {
        // single logical block; parallelize the flat range
        const dim_t total = block;
        parallel(0, [&](int ithr, int nthr) {
            dim_t start = 0, end = 0;
            balance211(total, nthr, ithr, start, end);
            if (start >= end) return;
            jit_uni_binary_kernel_t::call_params_t p;
            p.src0 = s0 + start * es;
            // scalar: same src1[0]; none: src1 advances with dst
            p.src1 = (bc == bcast_t::scalar) ? s1 : s1 + start * es;
            p.src2 = s2 ? s2 + start : nullptr; // s8 mask, 1 byte/elem
            p.dst = d + start * es;
            p.len = end - start;
            (*kernel_)(&p);
        });
    } else {
        // per-oc: one kernel call per block (channel block or pixel)
        parallel(0, [&](int ithr, int nthr) {
            dim_t bstart = 0, bend = 0;
            balance211(nb, nthr, ithr, bstart, bend);
            for (dim_t b = bstart; b < bend; b++) {
                jit_uni_binary_kernel_t::call_params_t p;
                p.src0 = s0 + b * block * es;
                p.dst = d + b * block * es;
                if (bc == bcast_t::per_oc_blocked)
                    p.src1 = s1 + (b % C) * es; // scalar = channel value
                else // per_oc_inner: per-element C-vector, same each pixel
                    p.src1 = s1;
                p.src2 = nullptr; // select never uses per-oc broadcast
                p.len = block;
                (*kernel_)(&p);
            }
        });
    }
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
