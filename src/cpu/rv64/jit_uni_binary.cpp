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
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

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

// Config baked into the kernel (broadcast/dtype/attr shape is compile-time; the
// scale/sum values and pointers arrive at run time via call_params_t).
struct binary_kernel_conf_t {
    alg_kind_t alg;
    bool scalar_src1; // src1 broadcasts the vectorized (inner) dim
    data_type_t dt_s0, dt_s1, dt_dst;
    bool do_scale0, do_scale1, do_sum;
    bool has_binary_po; // chain contains a binary post-op (advance rhs off)
    // src1 inner-dim element stride: 1 = contiguous (vle); >1 = strided (vlse,
    // src0/src1 different layouts, e.g. nchw:nhwc); ignored when scalar_src1.
    dim_t s1_inner_stride;
    bool is_select; // ternary select: dst = src2 ? src0 : src1 (src2 is s8)
    post_ops_t post_ops; // full post-op chain (sums applied in attribute order)
    const memory_desc_t *dst_md = nullptr; // for per_oc binary post-op rhs
    // Emit the unrolled (three src0/src1 pairs per step) f32 main loop; the
    // host enables it only for the streaming f32 case it applies to.
    bool unrolled = false;
    // Emit the narrow (single m1-vector, strictly interleaved) f32 main loop
    // instead — same streaming gate, selected per call at low thread counts.
    bool narrow = false;
};

struct jit_uni_binary_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src0;
        const void *src1;
        const void *src2; // ternary select condition (s8), else null
        const void *dst;
        const void *const *rhs_ptrs; // binary post-op rhs base array, or null
        dim_t len;
        size_t rhs_off; // element offset of the first lane for binary post-op rhs
        float scale0, scale1, sum_scale;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binary_kernel_t)

    jit_uni_binary_kernel_t(const binary_kernel_conf_t &c)
        : jit_generator_t("jit_uni_binary"), c_(c) {}
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

#define GET_OFF(f) (int)offsetof(call_params_t, f)
    void generate() override {
        ld(a1, a0, GET_OFF(src0));
        ld(a2, a0, GET_OFF(src1));
        ld(a5, a0, GET_OFF(src2));
        ld(a3, a0, GET_OFF(dst));
        ld(a6, a0, GET_OFF(rhs_ptrs));
        ld(a4, a0, GET_OFF(len));
        ld(a7, a0, GET_OFF(rhs_off));
        flw(fa4, a0, GET_OFF(scale0));
        flw(fa5, a0, GET_OFF(scale1));
        flw(fa6, a0, GET_OFF(sum_scale));
        emit_binary(); // handles the ordinary ops and ternary select
        ret();
    }
#undef GET_OFF

private:
    const binary_kernel_conf_t c_;

    // Registers (e32/m4 compute): a1=src0 a2=src1 a3=dst a4=len a5=src2
    // a6=rhs_ptrs a7=rhs_off(elems); t0=vl t1=bytes t2/t3=scratch; t4/t5/t6
    // hold the unrolled-loop strides (f32 bytes / step elements / stream
    // step); v0=mask
    // v4=acc v8=src1f32 v12/v16=narrow staging v20=binary-rhs; fa2=const
    // fa3=scalar-src1 fa4/fa5=scales fa6=sum fa7=binary-rhs-scalar
    // fa0/fa1=eltwise injector scratch; ft0/ft1=compare-result constants.
    const Reg p0 = a1, p1 = a2, pd = a3, len = a4, p2 = a5, rhs_ptrs = a6,
              off = a7, vl = t0, bytes = t1, gpr = t2;
    const VReg vd {4}, vs1 {8}, st0 {12}, st1 {16}, vrhs {20};
    const FReg f_s1 = fa3, f_sc0 = fa4, f_sc1 = fa5, f_sum = fa6, fc = fa2,
               f_zero = ft0, f_one = ft1;

    void setf(const FReg &f, float val) {
        uint32_t b;
        std::memcpy(&b, &val, sizeof(b));
        li(gpr, b);
        fmv_w_x(f, gpr);
    }

    // Set vl for `len` elements at the e32/m4 compute vtype.
    void set_compute_vl() {
        vsetvli(vl, len, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    }
    void to_compute_vtype() {
        vsetvli(x0, vl, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    }

    // Load `vl` elements of dtype dt from `ptr` into f32 group vf (e32/m4),
    // using `stage` for narrow staging. Ends at the e32/m4 vtype. stride_elems>1
    // reads with an element stride (vlse) — src0/src1 different plain layouts.
    void load_vec(data_type_t dt, const VReg &vf, const VReg &stage,
            const Reg &ptr, dim_t stride_elems = 1) {
        using namespace data_type;
        const bool strided = stride_elems > 1;
        if (strided) li(t3, stride_elems * (int)types::data_type_size(dt));
        auto ld = [&](int w, const VReg &v) {
            if (w == 32)
                strided ? vlse32_v(v, ptr, t3) : vle32_v(v, ptr);
            else if (w == 16)
                strided ? vlse16_v(v, ptr, t3) : vle16_v(v, ptr);
            else
                strided ? vlse8_v(v, ptr, t3) : vle8_v(v, ptr);
        };
        if (dt == f32) {
            to_compute_vtype();
            ld(32, vf);
        } else if (dt == s32) {
            to_compute_vtype();
            ld(32, vf);
            vfcvt_f_x_v(vf, vf);
        } else if (dt == f16) {
            vsetvli(x0, vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
            ld(16, stage);
            vfwcvt_f_f_v(vf, stage); // e16m2 -> e32m4
            to_compute_vtype();
        } else { // s8 / u8
            vsetvli(x0, vl, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
            ld(8, stage);
            to_compute_vtype();
            if (dt == s8) {
                vsext_vf4(vf, stage);
                vfcvt_f_x_v(vf, vf);
            } else {
                vzext_vf4(vf, stage);
                vfcvt_f_xu_v(vf, vf);
            }
        }
    }

    // Load one scalar of dtype dt from ptr as f32 into FReg f.
    void load_scalar(data_type_t dt, const FReg &f, const Reg &ptr) {
        using namespace data_type;
        if (dt == f32)
            flw(f, ptr, 0);
        else if (dt == s32) {
            lw(gpr, ptr, 0);
            fcvt_s_w(f, gpr);
        } else if (dt == s8) {
            lb(gpr, ptr, 0);
            fcvt_s_w(f, gpr);
        } else if (dt == u8) {
            lbu(gpr, ptr, 0);
            fcvt_s_wu(f, gpr);
        } else { // f16: widen via the vector unit, extract lane 0
            li(gpr, 1);
            vsetvli(x0, gpr, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
            vle16_v(vrhs, ptr);
            vfwcvt_f_f_v(vs1, vrhs); // e16m1 -> e32m2
            vsetvli(x0, gpr, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
            vfmv_f_s(f, vs1);
        }
    }

    // Convert f32 result group vd (e32/m4) to dt and store `vl` elements to ptr.
    void store_vec(
            data_type_t dt, const VReg &v, const VReg &stage, const Reg &ptr) {
        using namespace data_type;
        if (dt == f32) {
            to_compute_vtype();
            vse32_v(v, ptr);
        } else if (dt == s32) {
            setf(fc, -2147483648.0f);
            vfmax_vf(v, v, fc);
            // RISC-V vfcvt.x.f saturates; max f32 below 2^31 is 2147483520.
            setf(fc, 2147483520.0f);
            vfmin_vf(v, v, fc);
            vfcvt_x_f_v(v, v);
            vse32_v(v, ptr);
        } else if (dt == f16) {
            vsetvli(x0, vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
            vfncvt_f_f_w(stage, v); // e32m4 -> e16m2
            vse16_v(stage, ptr);
        } else { // s8 / u8
            const bool is_s8 = dt == s8;
            setf(fc, is_s8 ? -128.0f : 0.0f);
            vfmax_vf(v, v, fc);
            setf(fc, is_s8 ? 127.0f : 255.0f);
            vfmin_vf(v, v, fc);
            if (is_s8)
                vfcvt_x_f_v(v, v);
            else
                vfcvt_xu_f_v(v, v);
            vsetvli(x0, vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
            vnsrl_wi(stage, v, 0); // e32m4 -> e16m2
            vsetvli(x0, vl, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
            vnsrl_wi(st0, stage, 0); // e16m2 -> e8m1 (st0 as byte temp)
            vse8_v(st0, ptr);
        }
    }

    // v(acc) = v OP src1, comparisons yield 0.0/1.0. vtype is e32/m4.
    // f_zero/f_one are prepared once per kernel (see emit_binary).
    void cmp_to_01(const VReg &v) {
        vfmv_v_f(v, f_zero);
        vfmerge_vfm(v, v, f_one); // v[i] = v0[i] ? 1.0 : 0.0
    }
    // scratch takes the scalar-broadcast src1 of the max/min algorithms; it is
    // a dead vector group of the caller (vs1 in the generic loop).
    void binop_vf(const VReg &v, const FReg &f, const VReg &scratch) {
        using namespace alg_kind;
        switch (c_.alg) {
            case binary_add: vfadd_vf(v, v, f); break;
            case binary_sub: vfsub_vf(v, v, f); break;
            case binary_mul: vfmul_vf(v, v, f); break;
            case binary_div: vfdiv_vf(v, v, f); break;
            case binary_max:
                // nstl::max(src0,src1) = (src1 < src0) ? src0 : src1 (picks src1
                // on ties/unordered, matching the reference and x86 vmaxps).
                vfmv_v_f(scratch, f);
                vmflt_vv(VReg(0), scratch, v);
                vmerge_vvm(v, scratch, v);
                break;
            case binary_min:
                // nstl::min(src0,src1) = (src0 < src1) ? src0 : src1.
                vfmv_v_f(scratch, f);
                vmflt_vv(VReg(0), v, scratch);
                vmerge_vvm(v, scratch, v);
                break;
            case binary_ge:
                vmfge_vf(VReg(0), v, f);
                cmp_to_01(v);
                break;
            case binary_gt:
                vmfgt_vf(VReg(0), v, f);
                cmp_to_01(v);
                break;
            case binary_le:
                vmfle_vf(VReg(0), v, f);
                cmp_to_01(v);
                break;
            case binary_lt:
                vmflt_vf(VReg(0), v, f);
                cmp_to_01(v);
                break;
            case binary_eq:
                vmfeq_vf(VReg(0), v, f);
                cmp_to_01(v);
                break;
            case binary_ne:
                vmfne_vf(VReg(0), v, f);
                cmp_to_01(v);
                break;
            default: break;
        }
    }
    void binop_vv(const VReg &v, const VReg &v1) {
        using namespace alg_kind;
        switch (c_.alg) {
            case binary_add: vfadd_vv(v, v, v1); break;
            case binary_sub: vfsub_vv(v, v, v1); break;
            case binary_mul: vfmul_vv(v, v, v1); break;
            case binary_div: vfdiv_vv(v, v, v1); break;
            case binary_max:
                // nstl::max(src0,src1) = (src1 < src0) ? src0 : src1 (picks src1
                // on ties/unordered, matching the reference and x86 vmaxps).
                vmflt_vv(VReg(0), v1, v);
                vmerge_vvm(v, v1, v);
                break;
            case binary_min:
                // nstl::min(src0,src1) = (src0 < src1) ? src0 : src1.
                vmflt_vv(VReg(0), v, v1);
                vmerge_vvm(v, v1, v);
                break;
            // vmfgt/vmfge have no vv form: swap operands.
            case binary_ge:
                vmfle_vv(VReg(0), v1, v);
                cmp_to_01(v);
                break;
            case binary_gt:
                vmflt_vv(VReg(0), v1, v);
                cmp_to_01(v);
                break;
            case binary_le:
                vmfle_vv(VReg(0), v, v1);
                cmp_to_01(v);
                break;
            case binary_lt:
                vmflt_vv(VReg(0), v, v1);
                cmp_to_01(v);
                break;
            case binary_eq:
                vmfeq_vv(VReg(0), v, v1);
                cmp_to_01(v);
                break;
            case binary_ne:
                vmfne_vv(VReg(0), v, v1);
                cmp_to_01(v);
                break;
            default: break;
        }
    }

    // Ternary select condition: set v0 = (src2 == 0), reading vl elements into
    // `stage`. src2 is s8 (forced by the common binary descriptor). Ends at the
    // e32/m4 compute vtype (the mask is valid across the SEW switch: same vl).
    void select_mask(const VReg &stage) {
        vsetvli(x0, vl, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
        vle8_v(stage, p2);
        vmseq_vx(VReg(0), stage, x0); // v0 = (src2 == 0)
        to_compute_vtype();
    }

    void emit_binary() {
        using namespace data_type;
        // Post-op injectors (eltwise + binary chain) have five aux groups
        // (v8/v12/v16 + v24/v28); v24/v28 are free at m4 (vd=v4,
        // vs1=v8, staging v12/v16, binary rhs v20), so log/soft_relu/gelu_erf
        // eltwise post-ops fit here too.
        eltwise_injector::static_params_t esp(VReg(8), VReg(12), VReg(16),
                VReg(24), VReg(28), fa0, fa1, gpr, /*is_fwd=*/true);
        // Dynamic rhs addressing: the injector computes the rhs address from the
        // per-register output element offset (kept in `off`=a7 below) + the rhs
        // dtype/strategy in static params. `bytes` and t3 are address scratch.
        binary_injector::static_params_t bsp(vrhs, fa7, rhs_ptrs, bytes, t3);
        // default rhs dtype; the postops injector overrides it per binary from
        // src1_desc. Every rhs dtype is converted to f32 at load: scalar via
        // scalar loads (f16 via a stride-0 broadcast load), per_element and
        // per_oc/per_w gather via a narrow load + in-place widen.
        bsp.rhs_dt = f32;
        // per_oc gather index scratch (v24; the eltwise aux3 shares it but runs
        // in a different post-op entry, so there is no temporal overlap).
        bsp.v_idx = VReg(24);
        // narrow-rhs staging (v12 = st0, dead during post-ops; distinct from
        // v_rhs=v20 and v_idx=v24 so the widen into v_rhs is a legal overlap).
        bsp.v_tmp = VReg(12);
        // A select post-op loads its independent condition into v8. It reuses
        // the gather/narrow/address scratch after src1 has been materialized in
        // v20; all are dead from the main binary operation at this point.
        binary_injector::static_params_t select_bsp(
                VReg(8), fa0, rhs_ptrs, bytes, t3);
        select_bsp.v_idx = VReg(24);
        select_bsp.v_tmp = VReg(12);
        injector::jit_uni_postops_injector_t<v> po_inj(this, c_.post_ops, esp,
                c_.has_binary_po ? &bsp : nullptr, c_.dst_md, &select_bsp);
        binary_injector::rhs_arg_dynamic_params_t rhs_dyn;
        if (c_.has_binary_po) rhs_dyn.vmm_idx_to_out_off[vd.getIdx()] = off;

        // Preload the broadcast scalar src1 into f_s1 (f32), scaled once.
        if (c_.scalar_src1) {
            load_scalar(c_.dt_s1, f_s1, p1);
            if (c_.do_scale1) fmul_s(f_s1, f_s1, f_sc1);
        }

        // Comparison results are 0.0/1.0: the constants live in f_zero/f_one,
        // which no injector touches, so they are set once here instead of per
        // vector (see cmp_to_01).
        using namespace alg_kind;
        if (utils::one_of(c_.alg, binary_ge, binary_gt, binary_le, binary_lt,
                    binary_eq, binary_ne)) {
            fmv_w_x(f_zero, x0);
            setf(f_one, 1.0f);
        }

        Label loop, done;

        // Tuned f32 main loops (second kernels; the driver picks per call by
        // the thread count). Both cover only chains whose post-ops need no
        // injector aux (none, or only zero-alpha relu) and a contiguous f32
        // src1, so a zero-aux chain never reads the aux slots and they may
        // all alias the one group the loops leave unused.
        if (c_.unrolled || c_.narrow) {
            // All five eltwise aux vector groups are aliased to VReg(28):
            // the init-time gate below restricts the chain to empty or
            // zero-alpha relu, neither of which reads any aux vector group,
            // so a single group can be shared. Any future relaxation of
            // that gate must break this aliasing before forward-aux
            // algorithms become visible here.
            eltwise_injector::static_params_t esp_u(VReg(28), VReg(28),
                    VReg(28), VReg(28), VReg(28), fa0, fa1, gpr,
                    /*is_fwd=*/true);
            injector::jit_uni_postops_injector_t<v> po_u(
                    this, c_.post_ops, esp_u);
            binary_injector::rhs_arg_dynamic_params_t no_rhs;

            if (c_.unrolled) {
                // Three independent src0/src1 pairs keep six full-width loads
                // in flight. The loop needs the six m4 data groups. Fastest
                // when enough threads share the memory system.
                const VReg vdu[3] = {VReg(4), VReg(12), VReg(20)};
                const VReg vsu[3] = {VReg(8), VReg(16), VReg(24)};
                const Reg b = t4; // vl * 4 bytes (f32 element stride)
                const Reg n3 = t5; // 3 * vl elements per step
                const Reg b3 = t6; // 3 * vl * 4 bytes (stream step)
                const Reg a1r = t1, a2r = t2; // step-1/step-2 addresses
                vsetvli(vl, x0, SEW::e32, LMUL::m4, VTA::ta,
                        VMA::ma); // vl = VLMAX
                slli(b, vl, 2);
                slli(n3, vl, 1);
                add(n3, n3, vl);
                slli(b3, b, 1);
                add(b3, b3, b);
                Label main_loop, tail;
                L(main_loop);
                bltu(len, n3, tail);
                add(a1r, p0, b);
                add(a2r, a1r, b);
                vle32_v(vdu[0], p0);
                vle32_v(vdu[1], a1r);
                vle32_v(vdu[2], a2r);
                add(p0, p0, b3);
                if (!c_.scalar_src1) {
                    add(a1r, p1, b);
                    add(a2r, a1r, b);
                    vle32_v(vsu[0], p1);
                    vle32_v(vsu[1], a1r);
                    vle32_v(vsu[2], a2r);
                    add(p1, p1, b3);
                }
                for (int i = 0; i < 3; i++) {
                    if (c_.do_scale0) vfmul_vf(vdu[i], vdu[i], f_sc0);
                    if (!c_.scalar_src1 && c_.do_scale1)
                        vfmul_vf(vsu[i], vsu[i], f_sc1);
                    if (c_.scalar_src1)
                        binop_vf(vdu[i], f_s1, vsu[0]);
                    else
                        binop_vv(vdu[i], vsu[i]);
                    po_u.compute_vector(vdu[i].getIdx(), no_rhs);
                }
                add(a1r, pd, b);
                add(a2r, a1r, b);
                vse32_v(vdu[0], pd);
                vse32_v(vdu[1], a1r);
                vse32_v(vdu[2], a2r);
                add(pd, pd, b3);
                sub(len, len, n3);
                j_(main_loop);
                L(tail);
            } else {
                // Strictly interleaved single m1-vector loop with a hoisted
                // vsetvli: on an in-order core the vector queue drains in
                // order, so per-access width — not the number of groups —
                // sets the DRAM-visible request pattern, and the narrow
                // pattern extracts the most memory throughput when few
                // threads drive the memory system.
                const Reg b = t4; // vl * 4 bytes (f32 element stride)
                vsetvli(vl, x0, SEW::e32, LMUL::m1, VTA::ta,
                        VMA::ma); // vl = VLMAX
                slli(b, vl, 2);
                Label narrow_loop;
                L(narrow_loop);
                bltu(len, vl, loop);
                vle32_v(vd, p0);
                if (!c_.scalar_src1) vle32_v(vs1, p1);
                if (c_.do_scale0) vfmul_vf(vd, vd, f_sc0);
                if (!c_.scalar_src1 && c_.do_scale1) vfmul_vf(vs1, vs1, f_sc1);
                if (c_.scalar_src1)
                    binop_vf(vd, f_s1, vs1);
                else
                    binop_vv(vd, vs1);
                po_u.compute_vector(vd.getIdx(), no_rhs);
                vse32_v(vd, pd);
                add(p0, p0, b);
                if (!c_.scalar_src1) add(p1, p1, b);
                add(pd, pd, b);
                sub(len, len, vl);
                j_(narrow_loop);
            }
        }

        L(loop);
        beqz(len, done);

        set_compute_vl();
        load_vec(c_.dt_s0, vd, st0, p0); // src0 -> vd (f32)
        if (c_.do_scale0) vfmul_vf(vd, vd, f_sc0);
        if (!c_.scalar_src1) {
            // src1 -> vs1 (f32); strided for different plain layouts.
            load_vec(c_.dt_s1, vs1, st1, p1, c_.s1_inner_stride);
            if (c_.do_scale1) vfmul_vf(vs1, vs1, f_sc1);
        }

        if (c_.is_select) {
            // dst = src2 ? src0 : src1. Materialize a scalar-broadcast src1, then
            // v0 = (src2 == 0) and merge (v0 ? src1 : src0).
            if (c_.scalar_src1) vfmv_v_f(vs1, f_s1);
            select_mask(st1); // sets v0 = (src2 == 0), ends at e32/m4
            vmerge_vvm(vd, vd, vs1);
        } else if (c_.scalar_src1) {
            binop_vf(vd, f_s1, vs1);
        } else {
            binop_vv(vd, vs1);
        }

        // Apply the post-op chain in attribute order. Every sum entry invokes
        // the same old-dst lambda (the PD requires identical sum parameters),
        // matching the x64/AArch64 post-op injector contract.
        const int n_po = c_.post_ops.len();
        if (c_.do_sum) {
            int begin = 0;
            for (int i = 0; i < n_po; i++) {
                if (!c_.post_ops.entry_[i].is_sum(false, false)) continue;
                po_inj.compute_vector(vd.getIdx(), rhs_dyn, begin, i);
                load_vec(c_.dt_dst, st1, st0, pd); // old dst -> st1 (f32)
                vfmacc_vf(vd, f_sum, st1);
                begin = i + 1;
            }
            po_inj.compute_vector(vd.getIdx(), rhs_dyn, begin, n_po);
        } else {
            po_inj.compute_vector(vd.getIdx(), rhs_dyn, 0, n_po);
        }

        store_vec(c_.dt_dst, vd, st1, pd);

        // advance pointers by vl elements
        auto adv = [&](const Reg &p, data_type_t dt) {
            const int es = (int)types::data_type_size(dt);
            const int sh = es == 4 ? 2 : (es == 2 ? 1 : 0);
            if (sh)
                slli(bytes, vl, sh);
            else
                mv(bytes, vl);
            add(p, p, bytes);
        };
        // strided src1: advance by vl * stride * sizeof(dt).
        auto adv_strided = [&](const Reg &p, data_type_t dt, dim_t stride) {
            li(t3, stride * (int)types::data_type_size(dt));
            mul(bytes, vl, t3);
            add(p, p, bytes);
        };
        adv(p0, c_.dt_s0);
        adv(pd, c_.dt_dst);
        if (!c_.scalar_src1) {
            if (c_.s1_inner_stride > 1)
                adv_strided(p1, c_.dt_s1, c_.s1_inner_stride);
            else
                adv(p1, c_.dt_s1);
        }
        if (c_.is_select) adv(p2, data_type::s8); // src2 full, s8 (1 B/elem)
        // advance the rhs output ELEMENT offset by vl (the injector scales by
        // the rhs dtype size).
        if (c_.has_binary_po) add(off, off, vl);
        sub(len, len, vl);
        j_(loop);
        L(done);
    }
};

jit_uni_binary_t::jit_uni_binary_t(const pd_t *apd) : primitive_t(apd) {}
jit_uni_binary_t::~jit_uni_binary_t() = default;

status_t jit_uni_binary_t::init(engine_t *engine) {
    UNUSED(engine);
    const auto *p = pd();
    binary_kernel_conf_t c;
    c.alg = p->desc()->alg_kind;
    c.scalar_src1 = p->scalar_whole_ || p->scalar_inner_;
    c.dt_s0 = p->src_md(0)->data_type;
    c.dt_s1 = p->src_md(1)->data_type;
    c.dt_dst = p->dst_md()->data_type;
    c.do_scale0 = p->do_scale0_;
    c.do_scale1 = p->do_scale1_;
    c.do_sum = p->do_sum_;
    c.s1_inner_stride = p->s1_inner_stride_;
    c.is_select = p->desc()->alg_kind == alg_kind::binary_select;
    c.post_ops = p->po_;
    c.has_binary_po = c.post_ops.find(primitive_kind::binary) != -1;
    c.dst_md = p->dst_md(); // per_oc binary post-op rhs classification
    kernel_.reset(new jit_uni_binary_kernel_t(c));
    status_t status = kernel_->create_kernel();
    if (status != status::success) return status;

    // The tuned f32 streaming kernels (unrolled + narrow): the unrolled loop
    // needs the six m4 data groups, so they cover only chains whose post-ops
    // use no injector aux (none, or just zero-alpha relu) and a contiguous
    // f32 src1.
    const bool unrolled_ok = !c.is_select && !c.do_sum && !c.has_binary_po
            && c.s1_inner_stride == 1 && c.dt_s0 == data_type::f32
            && c.dt_s1 == data_type::f32 && c.dt_dst == data_type::f32
            && std::all_of(c.post_ops.entry_.begin(), c.post_ops.entry_.end(),
                    [](const post_ops_t::entry_t &e) {
        return e.is_eltwise() && e.eltwise.alg == alg_kind::eltwise_relu
                && e.eltwise.alpha == 0.f;
    });
    if (unrolled_ok) {
        // Both tuned kernels alias all five eltwise aux slots to VReg(28):
        // their post-op chain is empty or only zero-alpha relu, which never
        // reads the aux vector groups (relu_zero_ns uses only f_aux0 + vmask).
        // The 6 m4 data groups used by the unrolled loop likewise do not
        // collide with this aliasing.
        c.unrolled = true;
        kernel_unrolled_.reset(new jit_uni_binary_kernel_t(c));
        status = kernel_unrolled_->create_kernel();
        if (status != status::success) return status;
        c.unrolled = false;
        c.narrow = true;
        kernel_narrow_.reset(new jit_uni_binary_kernel_t(c));
        status = kernel_narrow_->create_kernel();
    }
    return status;
}

// Collect the per-binary rhs base pointers (post-op src1), advanced by their own
// logical origin, matching the x64/aarch64 post_ops_binary_rhs_arg_vec scheme.
static std::vector<const void *> collect_binary_rhs(
        const post_ops_t &po, const exec_ctx_t &ctx) {
    std::vector<const void *> rhs;
    for (int i = 0; i < po.len(); i++)
        if (po.entry_[i].is_binary()) {
            const memory_desc_wrapper s1_d(po.entry_[i].binary.src1_desc);
            const auto *base = static_cast<const char *>(ctx.host_ptr(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1));
            rhs.push_back(base + s1_d.off_l(0) * s1_d.data_type_size());
            if (po.entry_[i].is_binary_with_ternary_op()) {
                const memory_desc_wrapper s2_d(po.entry_[i].binary.src2_desc);
                const auto *base2 = static_cast<const char *>(ctx.host_ptr(
                        DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_2));
                rhs.push_back(base2 + s2_d.off_l(0) * s2_d.data_type_size());
            }
        }
    return rhs;
}

status_t jit_uni_binary_t::execute(const exec_ctx_t &ctx) const {
    const auto *pp = pd();
    const auto *s0 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_0);
    const auto *s1 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_1);
    auto *d = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const data_type_t dt_dst = pp->dst_md()->data_type;
    const data_type_t dt_s0 = pp->src_md(0)->data_type;
    const data_type_t dt_s1 = pp->src_md(1)->data_type;
    const size_t es0 = types::data_type_size(dt_s0);
    const size_t es1 = types::data_type_size(dt_s1);
    const size_t esd = types::data_type_size(dt_dst);

    // scale values (per-tensor, mask 0)
    float scale0 = 1.f, scale1 = 1.f;
    if (pp->do_scale0_)
        scale0 = *CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0);
    if (pp->do_scale1_)
        scale1 = *CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1);
    const float sum_scale = pp->sum_scale_;

    std::vector<const void *> po_rhs = collect_binary_rhs(pp->po_, ctx);
    const void *const *rhs_arr = po_rhs.empty() ? nullptr : po_rhs.data();

    // Honor md offsets (dense submemory views).
    s0 += (size_t)pp->src_md(0)->offset0 * es0;
    s1 += (size_t)pp->src_md(1)->offset0 * es1;
    d += (size_t)pp->dst_md()->offset0 * esd;

    // Ternary select reads a full (dst-shaped) src2 condition, advancing with
    // the run like src0/dst (the kernel folds select into the general path). s2
    // is null for non-select.
    const bool is_select = pp->desc()->alg_kind == alg_kind::binary_select;
    const char *s2 = nullptr;
    size_t es2 = 0;
    if (is_select) {
        s2 = CTX_IN_MEM(const char *, DNNL_ARG_SRC_2);
        es2 = types::data_type_size(pp->src_md(2)->data_type);
        s2 += (size_t)pp->src_md(2)->offset0 * es2;
    }

    auto fill = [&](jit_uni_binary_kernel_t::call_params_t &cp) {
        cp.scale0 = scale0;
        cp.scale1 = scale1;
        cp.sum_scale = sum_scale;
        cp.rhs_ptrs = rhs_arr;
        cp.src2 = nullptr;
    };
    // The tuned kernels cover the streaming f32 case. The unrolled kernel
    // issues per-stream bursts of loads/stores, which extracts more memory
    // throughput once enough threads share the memory system; below that the
    // narrow single-vector interleave is fastest, so pick per call by the
    // thread count.
    auto pick_kernel = [&](int nthr) -> const jit_uni_binary_kernel_t & {
        if (nthr >= unrolled_kernel_min_nthrs && kernel_unrolled_)
            return *kernel_unrolled_;
        if (kernel_narrow_) return *kernel_narrow_;
        return *kernel_;
    };

    if (pp->whole_) {
        // one flat pass; parallelize the range. src1 is scalar or matches dst.
        const dim_t total = pp->total_;
        const bool scalar = pp->scalar_whole_;
        parallel(0, [&](int ithr, int nthr) {
            dim_t start = 0, end = 0;
            balance211(total, nthr, ithr, start, end);
            if (start >= end) return;
            jit_uni_binary_kernel_t::call_params_t cp = {};
            fill(cp);
            cp.src0 = s0 + start * es0;
            cp.src1 = scalar ? s1 : s1 + start * es1;
            if (is_select) cp.src2 = s2 + start * es2;
            cp.dst = d + start * esd;
            cp.len = end - start;
            cp.rhs_off = (size_t)start; // element offset (injector scales)
            pick_kernel(nthr)(&cp);
        });
        return status::success;
    }

    // General broadcast: iterate runs of `inner` elements; offset src1 by the
    // per-dim broadcast strides. The inner dim is either contiguous in src1
    // (vector) or broadcast (scalar), captured by pp->scalar_inner_.
    const dim_t inner = pp->inner_;
    const dim_t n_outer = pp->n_outer_;
    const int nd = pp->nd_;
    parallel(0, [&](int ithr, int nthr) {
        dim_t bstart = 0, bend = 0;
        balance211(n_outer, nthr, ithr, bstart, bend);
        for (dim_t b = bstart; b < bend; b++) {
            // decompose the run index b over dst dims [0..nd-2] to compute the
            // src1 element offset via broadcast strides.
            dim_t s1_off = 0, rem = b;
            bool tail_run = false;
            for (int i = nd - 2; i >= 0; i--) {
                const dim_t dim = pp->out_dims_[i];
                const dim_t idx = rem % dim;
                rem /= dim;
                s1_off += idx * pp->s1_str_[i];
                if (i == pp->tail_axis_) tail_run = idx == dim - 1;
            }
            if (pp->s1_same_layout_) s1_off = b * inner;
            const dim_t run = tail_run ? pp->tail_ : inner;
            jit_uni_binary_kernel_t::call_params_t cp = {};
            fill(cp);
            cp.src0 = s0 + b * inner * es0;
            cp.src1 = s1 + s1_off * es1;
            if (is_select) cp.src2 = s2 + b * inner * es2;
            cp.dst = d + b * inner * esd;
            cp.len = run;
            cp.rhs_off = (size_t)b * inner; // element offset (injector scales)
            pick_kernel(nthr)(&cp);
            if (run < inner)
                std::memset(
                        d + (b * inner + run) * esd, 0, (inner - run) * esd);
        }
    });
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
