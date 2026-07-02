/*******************************************************************************
* Copyright 2026 openKylin community
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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/float16.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_uni_prelu.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

// Read one weights element of an arbitrary supported dtype as f32 (host side,
// used by the scalar-broadcast paths so the kernel only ever sees an f32).
float load_weight_f32(const void *base, dim_t idx, data_type_t dt) {
    switch (dt) {
        case data_type::f32: return reinterpret_cast<const float *>(base)[idx];
        case data_type::f16:
            return static_cast<float>(
                    reinterpret_cast<const float16_t *>(base)[idx]);
        case data_type::bf16:
            return static_cast<float>(
                    reinterpret_cast<const bfloat16_t *>(base)[idx]);
        default: assert(!"unsupported dtype"); return 0.f;
    }
}

// Emit a widening f32 convert of a 16-bit group: bf16 via Zvfbfmin, f16 via Zvfh.
void emit_widen16(
        jit_generator_t *h, const VReg &vd, const VReg &vs, data_type_t dt) {
    if (dt == data_type::bf16)
        h->vfwcvtbf16_f_f_v(vd, vs); // Zvfbfmin
    else
        h->vfwcvt_f_f_v(vd, vs); // Zvfh
}

// Emit one vector-length-agnostic PReLU pass over `len` elements:
//   dst = max(0, src) + weights * min(0, src)
// computed in f32. Weights are either a per-element vector that walks in
// lockstep with src (scalar_mode == false) or a single f32 splat (scalar_mode).
// src/dst share dtype `dt`; the lockstep weights carry their own `wei_dt`, so
// every src x weights dtype combo is supported. 16-bit types widen/narrow with
// the native FP converts (Zvfh vfwcvt/vfncvt, Zvfbfmin vfwcvtbf16/vfncvtbf16).
//
// Register layout: a1=src, a2=weights, a3=dst, a4=len; t0=vl, t1=bytes,
// t2=wei bytes; v8=f32 src/compute group, v12=max, v16=min (== dst),
// v20=weights vector, v4=16-bit load staging; fa0=zero const, fa1=scalar weight.
void emit_prelu_loop(jit_generator_t *h, data_type_t dt, data_type_t wei_dt,
        bool scalar_mode, size_t weight_off) {
    const Reg p_in = a1, p_w = a2, p_out = a3, len = a4, vl = t0, bytes = t1,
              wbytes = t2;
    const VReg vsrc(8), vmax(12), vmin(16), vw(20), vt(4);
    const FReg fzero = fa0, fw = fa1;

    const bool src16 = utils::one_of(dt, data_type::f16, data_type::bf16);
    const bool wei16 = utils::one_of(wei_dt, data_type::f16, data_type::bf16);
    // Compute LMUL: 16-bit src widens e16m1 -> e32m2, f32 src stays e32m1.
    const LMUL clmul = src16 ? LMUL::m2 : LMUL::m1;

    h->fmv_w_x(fzero, x0); // 0.0f
    if (scalar_mode) h->flw(fw, a0, static_cast<int>(weight_off));

    Label loop, done;
    h->L(loop);
    h->beqz(len, done);

    // All vsetvli use the tail/mask-agnostic policy (ta, ma): every iteration
    // processes exactly vl elements (no masking, no tail reads), so agnostic is
    // result-equivalent and avoids preserving inactive elements on OoO cores.
    // ---- load + widen src to f32 in vsrc; leaves vtype = e32, clmul ----
    if (!src16) {
        h->vsetvli(vl, len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        h->vle32_v(vsrc, p_in);
    } else {
        h->vsetvli(vl, len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        h->vle16_v(vt, p_in);
        emit_widen16(h, vsrc, vt, dt); // e16m1 -> e32m2
        h->vsetvli(x0, vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
    }

    // ---- load + widen weights to f32 in vw (lockstep only) ----
    if (!scalar_mode) {
        if (!wei16) {
            // f32 weights: load directly at the compute vtype (e32, clmul).
            h->vle32_v(vw, p_w);
        } else {
            // 16-bit weights load at half the compute LMUL, then widen.
            const LMUL wlmul = (clmul == LMUL::m1) ? LMUL::mf2 : LMUL::m1;
            h->vsetvli(x0, vl, SEW::e16, wlmul, VTA::ta, VMA::ma);
            h->vle16_v(vt, p_w);
            emit_widen16(h, vw, vt, wei_dt); // e16(wlmul) -> e32(clmul)
            h->vsetvli(x0, vl, SEW::e32, clmul, VTA::ta,
                    VMA::ma); // restore compute vtype
        }
    }

    // ---- compute: dst = max(0, src) + weights * min(0, src) ----
    h->vfmax_vf(vmax, vsrc, fzero);
    h->vfmin_vf(vmin, vsrc, fzero);
    if (scalar_mode)
        h->vfmadd_vf(vmin, fw, vmax); // vmin = vmin * fw + vmax
    else
        h->vfmadd_vv(vmin, vw, vmax); // vmin = vmin * vw + vmax
    // result now in vmin

    // ---- narrow + store (dst dtype == src dtype) ----
    if (!src16) {
        h->vse32_v(vmin, p_out);
    } else { // f16 / bf16: narrow f32 -> 16-bit (e32m2 -> e16m1) and store
        h->vsetvli(x0, vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        if (dt == data_type::bf16)
            h->vfncvtbf16_f_f_w(vt, vmin); // Zvfbfmin, round-to-nearest-even
        else
            h->vfncvt_f_f_w(vt, vmin); // Zvfh
        h->vse16_v(vt, p_out);
    }

    // ---- advance pointers (src/dst by dt size, weights by wei_dt size) ----
    h->slli(bytes, vl, src16 ? 1 : 2);
    h->add(p_in, p_in, bytes);
    h->add(p_out, p_out, bytes);
    if (!scalar_mode) {
        if (wei16 == src16) {
            h->add(p_w, p_w, bytes);
        } else {
            h->slli(wbytes, vl, wei16 ? 1 : 2);
            h->add(p_w, p_w, wbytes);
        }
    }
    h->sub(len, len, vl);
    h->j_(loop);

    h->L(done);
}

} // namespace

struct jit_uni_prelu_fwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        const void *weights; // lockstep mode only
        void *dst;
        dim_t len;
        float weight; // scalar mode only
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_prelu_fwd_kernel_t)

    jit_uni_prelu_fwd_kernel_t(
            data_type_t dt, data_type_t wei_dt, bool scalar_mode)
        : jit_generator_t("jit_uni_prelu_fwd")
        , dt_(dt)
        , wei_dt_(wei_dt)
        , scalar_mode_(scalar_mode) {
        create_kernel();
    }
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }
    void generate() override {
        ld(a1, a0, offsetof(call_params_t, src));
        if (!scalar_mode_) ld(a2, a0, offsetof(call_params_t, weights));
        ld(a3, a0, offsetof(call_params_t, dst));
        ld(a4, a0, offsetof(call_params_t, len));
        emit_prelu_loop(this, dt_, wei_dt_, scalar_mode_,
                offsetof(call_params_t, weight));
        ret();
    }
    data_type_t dt_;
    data_type_t wei_dt_;
    bool scalar_mode_;
};

jit_uni_prelu_fwd_t::jit_uni_prelu_fwd_t(const pd_t *apd) : primitive_t(apd) {}
jit_uni_prelu_fwd_t::~jit_uni_prelu_fwd_t() = default;

status_t jit_uni_prelu_fwd_t::init(engine_t *engine) {
    UNUSED(engine);
    const data_type_t dt = pd()->src_md(0)->data_type;
    const data_type_t wei_dt = pd()->weights_md(0)->data_type;
    // scalar / per_oc_nchw read the weight on the host (any dtype); the lockstep
    // paths (full / per_oc_nhwc / per_oc_blocked) load weights in-kernel.
    const bool needs_scalar = utils::one_of(
            pd()->bcast_, prelu_bcast_t::scalar, prelu_bcast_t::per_oc_nchw);
    if (needs_scalar)
        scalar_kernel_.reset(new jit_uni_prelu_fwd_kernel_t(
                dt, wei_dt, /*scalar_mode=*/true));
    else
        kernel_.reset(new jit_uni_prelu_fwd_kernel_t(
                dt, wei_dt, /*scalar_mode=*/false));
    return status::success;
}

status_t jit_uni_prelu_fwd_t::execute(const exec_ctx_t &ctx) const {
    using kparams_t = jit_uni_prelu_fwd_kernel_t::call_params_t;

    const auto *src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto *weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto *dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const data_type_t dt = pd()->src_md(0)->data_type;
    const data_type_t wei_dt = pd()->weights_md(0)->data_type;
    const size_t es = types::data_type_size(dt);
    const size_t wes = types::data_type_size(wei_dt);

    const memory_desc_wrapper src_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const dim_t nelems = src_d.nelems(true);
    const char *src_base = src + (size_t)src_d.offset0() * es;
    char *dst_base = dst + (size_t)src_d.offset0() * es;
    const char *wei_base = weights + (size_t)weights_d.offset0() * wes;

    const int max_thr = dnnl_get_max_threads();

    switch (pd()->bcast_) {
        case prelu_bcast_t::full: {
            // Flat lockstep over the whole tensor; src and weights share layout.
            parallel(max_thr, [&](int ithr, int nthr) {
                dim_t start = 0, end = 0;
                balance211(nelems, nthr, ithr, start, end);
                if (start >= end) return;
                kparams_t p;
                p.src = src_base + start * es;
                p.weights = wei_base + start * wes;
                p.dst = dst_base + start * es;
                p.len = end - start;
                (*kernel_)(&p);
            });
            break;
        }
        case prelu_bcast_t::scalar: {
            const float w = load_weight_f32(wei_base, 0, wei_dt);
            parallel(max_thr, [&](int ithr, int nthr) {
                dim_t start = 0, end = 0;
                balance211(nelems, nthr, ithr, start, end);
                if (start >= end) return;
                kparams_t p;
                p.src = src_base + start * es;
                p.dst = dst_base + start * es;
                p.len = end - start;
                p.weight = w;
                (*scalar_kernel_)(&p);
            });
            break;
        }
        case prelu_bcast_t::per_oc_nhwc: {
            // Channel is innermost: each contiguous run of C elements pairs with
            // weights[0..C). Parallelize over the N*spatial outer runs.
            const dim_t C = src_d.dims()[1];
            const dim_t outer = nelems / C;
            parallel(max_thr, [&](int ithr, int nthr) {
                dim_t start = 0, end = 0;
                balance211(outer, nthr, ithr, start, end);
                for (dim_t i = start; i < end; ++i) {
                    kparams_t p;
                    p.src = src_base + (size_t)i * C * es;
                    p.weights = wei_base;
                    p.dst = dst_base + (size_t)i * C * es;
                    p.len = C;
                    (*kernel_)(&p);
                }
            });
            break;
        }
        case prelu_bcast_t::per_oc_nchw: {
            // Channel-major plain: each (n, c) spatial plane of SP contiguous
            // elements shares one weight w[c]. Parallelize over the N*C planes.
            const dim_t N = src_d.dims()[0];
            const dim_t C = src_d.dims()[1];
            const dim_t planes = N * C;
            const dim_t SP = nelems / planes;
            parallel(max_thr, [&](int ithr, int nthr) {
                dim_t start = 0, end = 0;
                balance211(planes, nthr, ithr, start, end);
                for (dim_t pl = start; pl < end; ++pl) {
                    const dim_t c = pl % C;
                    kparams_t p;
                    p.src = src_base + (size_t)pl * SP * es;
                    p.dst = dst_base + (size_t)pl * SP * es;
                    p.len = SP;
                    p.weight = load_weight_f32(wei_base, c, wei_dt);
                    (*scalar_kernel_)(&p);
                }
            });
            break;
        }
        case prelu_bcast_t::per_oc_blocked: {
            // Channel-blocked (nChw{8,16}c): the layout is [N][C/blk][sp][blk],
            // so each contiguous run of `blk` elements is one channel block at
            // some (n, c_outer, sp) and pairs in lockstep with weights[c_outer
            // * blk .. +blk). Walk over those runs, parallelizing across them.
            const auto &bd = src_d.blocking_desc();
            const dim_t blk = bd.inner_blks[0];
            const dim_t N = src_d.padded_dims()[0];
            const dim_t Cpad = src_d.padded_dims()[1];
            const dim_t Cb = Cpad / blk;
            const dim_t HW
                    = nelems / (N * Cpad); // spatial positions per (n,co)
            const dim_t runs = nelems / blk;
            parallel(max_thr, [&](int ithr, int nthr) {
                dim_t start = 0, end = 0;
                balance211(runs, nthr, ithr, start, end);
                for (dim_t r = start; r < end; ++r) {
                    const dim_t c_outer = (r / HW) % Cb;
                    kparams_t p;
                    p.src = src_base + (size_t)r * blk * es;
                    p.weights = wei_base + (size_t)c_outer * blk * wes;
                    p.dst = dst_base + (size_t)r * blk * es;
                    p.len = blk;
                    (*kernel_)(&p);
                }
            });
            break;
        }
        default: assert(!"unsupported broadcast"); return status::runtime_error;
    }
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
