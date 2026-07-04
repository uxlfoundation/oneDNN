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

#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_uni_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

// Emit one vector-length-agnostic pass: load `len` elements from a1 (and, for
// backward, diff_dst from a3), convert to f32, run the injector, convert back
// (with saturation for int types), store to a2. The injector always computes in
// f32; non-f32 dtypes are converted (with saturation) at the load/store edge.
//
// Register layout (max LMUL = m4): a1=in, a2=out, a3=diff_dst, a4=len,
// t0=vl, t1=bytes, t2=gpr scratch; v4 = f32 compute group, v2 = 8/16-bit
// staging, v20 = diff_dst (bwd) reused as the i16 narrow temp, v8/v12/v16 =
// injector aux, v0 = bwd mask, fa0/fa1 = injector FP scratch, fa2 = clamp const.
void emit_eltwise_loop(jit_generator_t *h, alg_kind_t alg, float alpha,
        float beta, float scale, data_type_t dt, bool is_fwd) {
    const Reg p_in = a1, p_out = a2, p_dd = a3, len = a4, vl = t0, bytes = t1,
              gpr = t2;
    const VReg vd(4), vt(2), vdd(20), vi16(20);
    const FReg fc = fa2;

    eltwise_injector::static_params_t sp(
            VReg(8), VReg(12), VReg(16), fa0, fa1, gpr, is_fwd);
    jit_uni_eltwise_injector_t<v> inj(h, alg, alpha, beta, scale, sp);

    auto setf = [&](float val) {
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        h->li(gpr, bits);
        h->fmv_w_x(fc, gpr);
    };

    const bool is_s8 = dt == data_type::s8;

    Label loop, done;
    h->L(loop);
    h->beqz(len, done);

    // ---- load src (+ diff_dst) and widen to f32 in vd (and vdd) ----
    if (dt == data_type::f32) {
        h->vsetvli(vl, len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        h->vle32_v(vd, p_in);
        if (!is_fwd) h->vle32_v(vdd, p_dd);
    } else if (dt == data_type::s32) {
        h->vsetvli(vl, len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        h->vle32_v(vd, p_in);
        h->vfcvt_f_x_v(vd, vd);
        if (!is_fwd) {
            h->vle32_v(vdd, p_dd);
            h->vfcvt_f_x_v(vdd, vdd);
        }
    } else if (dt == data_type::f16) {
        h->vsetvli(vl, len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        h->vle16_v(vt, p_in);
        h->vfwcvt_f_f_v(vd, vt); // e16m1 -> e32m2
        if (!is_fwd) {
            h->vle16_v(vt, p_dd);
            h->vfwcvt_f_f_v(vdd, vt);
        }
        h->vsetvli(x0, vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
    } else { // s8 / u8
        const VReg vt2(3); // second 8-bit staging reg (backward diff_dst)
        h->vsetvli(vl, len, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
        h->vle8_v(vt, p_in);
        if (!is_fwd) h->vle8_v(vt2, p_dd);
        // vsext/vzext.vf4 operate at the DESTINATION vtype (e32m4) and read the
        // source group as 8-bit, so switch vtype before extending.
        h->vsetvli(x0, vl, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        if (is_s8) {
            h->vsext_vf4(vd, vt);
            if (!is_fwd) h->vsext_vf4(vdd, vt2);
            h->vfcvt_f_x_v(vd, vd);
            if (!is_fwd) h->vfcvt_f_x_v(vdd, vdd);
        } else {
            h->vzext_vf4(vd, vt);
            if (!is_fwd) h->vzext_vf4(vdd, vt2);
            h->vfcvt_f_xu_v(vd, vd);
            if (!is_fwd) h->vfcvt_f_xu_v(vdd, vdd);
        }
    }

    // ---- compute (forward: alg(s); backward: alg'(s) * diff_dst) ----
    inj.compute_vector(vd.getIdx());
    if (!is_fwd) h->vfmul_vv(vd, vd, vdd);

    // ---- convert back + store (vtype currently e32 at the compute LMUL) ----
    if (dt == data_type::f32) {
        h->vse32_v(vd, p_out);
    } else if (dt == data_type::s32) {
        setf(-2147483648.0f);
        h->vfmax_vf(vd, vd, fc);
        setf(2147483647.0f);
        h->vfmin_vf(vd, vd, fc);
        h->vfcvt_x_f_v(vd, vd);
        h->vse32_v(vd, p_out);
    } else if (dt == data_type::f16) {
        h->vsetvli(x0, vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        h->vfncvt_f_f_w(vt, vd); // e32m2 -> e16m1
        h->vse16_v(vt, p_out);
    } else { // s8 / u8
        setf(is_s8 ? -128.0f : 0.0f);
        h->vfmax_vf(vd, vd, fc);
        setf(is_s8 ? 127.0f : 255.0f);
        h->vfmin_vf(vd, vd, fc);
        if (is_s8)
            h->vfcvt_x_f_v(vd, vd);
        else
            h->vfcvt_xu_f_v(vd, vd);
        // narrow i32(m4) -> i16(m2) -> i8(m1); values pre-clamped so truncation
        // via vnsrl by 0 is exact.
        h->vsetvli(x0, vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
        h->vnsrl_wi(vi16, vd, 0);
        h->vsetvli(x0, vl, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
        h->vnsrl_wi(vt, vi16, 0);
        h->vse8_v(vt, p_out);
    }

    // ---- advance pointers by vl * sizeof(dtype) and loop ----
    const int dsz = static_cast<int>(types::data_type_size(dt));
    const int sh = dsz == 4 ? 2 : (dsz == 2 ? 1 : 0);
    if (sh)
        h->slli(bytes, vl, sh);
    else
        h->mv(bytes, vl);
    h->add(p_in, p_in, bytes);
    h->add(p_out, p_out, bytes);
    if (!is_fwd) h->add(p_dd, p_dd, bytes);
    h->sub(len, len, vl);
    h->j_(loop);

    h->L(done);
}

} // namespace

struct jit_uni_eltwise_fwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        void *dst;
        dim_t len;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_fwd_kernel_t)

    jit_uni_eltwise_fwd_kernel_t(alg_kind_t alg, float alpha, float beta,
            float scale, data_type_t dt)
        : jit_generator_t("jit_uni_eltwise_fwd")
        , alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , dt_(dt) {
        create_kernel();
    }
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }
    void generate() override {
        ld(a1, a0, 0); // src
        ld(a2, a0, 8); // dst
        ld(a4, a0, 16); // len
        emit_eltwise_loop(this, alg_, alpha_, beta_, scale_, dt_, true);
        ret();
    }
    alg_kind_t alg_;
    float alpha_, beta_, scale_;
    data_type_t dt_;
};

struct jit_uni_eltwise_bwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        const void *diff_dst;
        void *diff_src;
        dim_t len;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_bwd_kernel_t)

    jit_uni_eltwise_bwd_kernel_t(
            alg_kind_t alg, float alpha, float beta, data_type_t dt)
        : jit_generator_t("jit_uni_eltwise_bwd")
        , alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , dt_(dt) {
        create_kernel();
    }
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }
    void generate() override {
        ld(a1, a0, 0); // src
        ld(a3, a0, 8); // diff_dst
        ld(a2, a0, 16); // diff_src
        ld(a4, a0, 24); // len
        emit_eltwise_loop(this, alg_, alpha_, beta_, 1.f, dt_, false);
        ret();
    }
    alg_kind_t alg_;
    float alpha_, beta_;
    data_type_t dt_;
};

// ---- forward primitive ----
template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}
template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::~jit_uni_eltwise_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);
    const auto &d = *pd()->desc();
    kernel_.reset(new jit_uni_eltwise_fwd_kernel_t(
            d.alg_kind, d.alpha, d.beta, 1.f, pd()->dst_md()->data_type));
    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const memory_desc_wrapper data_d(pd()->src_md());
    const dim_t len = data_d.nelems(true);
    const size_t es = types::data_type_size(pd()->dst_md()->data_type);
    // Honor a non-zero md offset (dense submemory view); src and dst share the
    // same md here (the pd enforces src_d == dst_d).
    const size_t base = (size_t)data_d.offset0() * es;

    const int max_thr = dnnl_get_max_threads();
    parallel(max_thr, [&](int ithr, int nthr) {
        dim_t start = 0, end = 0;
        balance211(len, nthr, ithr, start, end);
        if (start >= end) return;
        jit_uni_eltwise_fwd_kernel_t::call_params_t p;
        p.src = static_cast<const char *>(src) + base + start * es;
        p.dst = static_cast<char *>(dst) + base + start * es;
        p.len = end - start;
        (*kernel_)(&p);
    });
    return status::success;
}

// ---- backward primitive ----
template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}
template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::~jit_uni_eltwise_bwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);
    const auto &d = *pd()->desc();
    kernel_.reset(new jit_uni_eltwise_bwd_kernel_t(
            d.alg_kind, d.alpha, d.beta, pd()->src_md()->data_type));
    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const auto *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto *diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
    const dim_t len = memory_desc_wrapper(pd()->diff_dst_md()).nelems(true);
    const size_t es = types::data_type_size(pd()->src_md()->data_type);
    // Honor non-zero md offsets (dense submemory views); each operand may carry
    // its own offset0.
    const size_t src_base
            = (size_t)memory_desc_wrapper(pd()->src_md()).offset0() * es;
    const size_t ddst_base
            = (size_t)memory_desc_wrapper(pd()->diff_dst_md()).offset0() * es;
    const size_t dsrc_base
            = (size_t)memory_desc_wrapper(pd()->diff_src_md()).offset0() * es;

    const int max_thr = dnnl_get_max_threads();
    parallel(max_thr, [&](int ithr, int nthr) {
        dim_t start = 0, end = 0;
        balance211(len, nthr, ithr, start, end);
        if (start >= end) return;
        jit_uni_eltwise_bwd_kernel_t::call_params_t p;
        p.src = static_cast<const char *>(src) + src_base + start * es;
        p.diff_dst
                = static_cast<const char *>(diff_dst) + ddst_base + start * es;
        p.diff_src = static_cast<char *>(diff_src) + dsrc_base + start * es;
        p.len = end - start;
        (*kernel_)(&p);
    });
    return status::success;
}

template struct jit_uni_eltwise_fwd_t<v>;
template struct jit_uni_eltwise_bwd_t<v>;
template struct jit_uni_eltwise_fwd_t<zvfh>;
template struct jit_uni_eltwise_bwd_t<zvfh>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
