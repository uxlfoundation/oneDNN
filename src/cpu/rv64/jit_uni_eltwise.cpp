/*******************************************************************************
* Copyright 2017 Intel Corporation
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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/rv64/jit_generator.hpp"

#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/rv64/jit_uni_eltwise.hpp"

#define GET_OFF(field) offsetof(jit_args_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

struct jit_args_t {
    const void *src; // fwd: src;  bwd: src/dst based on alg;
    const void *dst; // fwd: dst;  bwd: diff_src;
    const void *diff_dst; // fwd: nullptr;  bwd: diff_dst;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel_t : public jit_generator_t {
    jit_uni_eltwise_kernel_t(const eltwise_pd_t *pd)
        : jit_generator_t("jit_uni_eltwise_kernel"), pd_(pd) {}

    void operator()(const jit_args_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    const eltwise_pd_t *pd_;

    data_type_t data_type() const {
        return pd_->use_dst() ? pd_->dst_md()->data_type
                              : pd_->src_md()->data_type;
    }
    int dtype_size() const { return types::data_type_size(data_type()); }
};

// jit kernels
namespace {

// Vector-length-agnostic eltwise kernel. A single vsetvli loop covers the main
// body and the tail alike, so there is no unrolled/remainder loop split as on
// x64/aarch64, and no isa template: RVV has one scalable vector ISA and the
// generated code adapts to the hardware VLEN at run time.
//
// The injector always computes in f32: f16 (zvfh instance) is converted at the
// load/store edge, and the integer dtypes (s32/s8/u8, forward only, v
// instance) are widened to f32 on load and saturated back to the dst range on
// store -- the reference also evaluates eltwise in float, so no separate
// integer kernel is needed. RISC-V vfcvt.x.f saturates, so the upper s32 clamp
// is max_value<f32>(s32) == 2147483520.f (2147483647.f is not representable
// and rounds up to 2^31); the s8/u8 bounds are exact.
//
// Register layout: a1 = reg_src, a2 = reg_dst, a3 = reg_diff_dst,
// a4 = reg_work_amount, t0 = reg_vl, t1 = reg_bytes, t2 = reg_tmp;
// v4 = vmm_src (f32 compute group), v2 = vmm_tmp (f16/int staging),
// v20 = vmm_diff_dst (bwd), v8/v12/v16 = injector aux0..2, v0 = mask,
// fa0/fa1 = injector FP scratch (fa0 is reused to materialize the integer
// saturation bounds after the injector is done). The injector's 4th/5th aux
// (gelu_erf fwd, gelu_tanh bwd) are v20/v24 in forward (diff_dst absent) and
// v24/v28 in backward (v20 holds diff_dst). The compute LMUL is m1 for f32,
// m2 for f16 (e16/m1 widening pair), and m4 for s32 and s8/u8 (e8/m1 widening
// pair); every group used (v4, v8, v12, v16, v20, v24) is 4-aligned, so the
// layout is legal at each of these LMULs.
struct jit_uni_kernel_t : public jit_uni_eltwise_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel)

    jit_uni_kernel_t(const eltwise_pd_t *pd)
        : jit_uni_eltwise_kernel_t(pd), is_fwd_(pd_->is_fwd()) {
        const auto &desc = *pd_->desc();
        const VReg vmm_aux3 = is_fwd_ ? VReg(20) : VReg(24);
        const VReg vmm_aux4 = is_fwd_ ? VReg(24) : VReg(28);
        eltwise_injector::static_params_t sp(VReg(8), VReg(12), VReg(16),
                vmm_aux3, vmm_aux4, fa0, fa1, reg_tmp, is_fwd_);
        eltwise_injector_.reset(new jit_uni_eltwise_injector_t<v>(
                this, desc.alg_kind, desc.alpha, desc.beta, 1.f, sp));
    }

    // Load `vl` src (and, for backward, diff_dst) elements and widen them to
    // f32 in vmm_src (and vmm_diff_dst). Issues the vsetvli that sets reg_vl
    // and leaves the vtype at e32 / the dtype's compute LMUL for the injector.
    void load_vector() {
        const data_type_t dt = data_type();
        if (dt == data_type::f32) {
            vsetvli(reg_vl, reg_work_amount, SEW::e32, LMUL::m1, VTA::ta,
                    VMA::ma);
            vle32_v(vmm_src, reg_src);
            if (!is_fwd_) vle32_v(vmm_diff_dst, reg_diff_dst);
        } else if (dt == data_type::f16) {
            vsetvli(reg_vl, reg_work_amount, SEW::e16, LMUL::m1, VTA::ta,
                    VMA::ma);
            vle16_v(vmm_tmp, reg_src);
            vfwcvt_f_f_v(vmm_src, vmm_tmp); // e16m1 -> e32m2
            if (!is_fwd_) {
                vle16_v(vmm_tmp, reg_diff_dst);
                vfwcvt_f_f_v(vmm_diff_dst, vmm_tmp);
            }
            vsetvli(x0, reg_vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        } else if (dt == data_type::s32) {
            vsetvli(reg_vl, reg_work_amount, SEW::e32, LMUL::m4, VTA::ta,
                    VMA::ma);
            vle32_v(vmm_src, reg_src);
            vfcvt_f_x_v(vmm_src, vmm_src);
        } else { // s8 / u8
            vsetvli(reg_vl, reg_work_amount, SEW::e8, LMUL::m1, VTA::ta,
                    VMA::ma);
            vle8_v(vmm_tmp, reg_src);
            // vsext/vzext.vf4 operate at the destination vtype (e32/m4) and
            // read the source group as 8-bit, so switch vtype before extending.
            vsetvli(x0, reg_vl, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
            if (dt == data_type::s8) {
                vsext_vf4(vmm_src, vmm_tmp);
                vfcvt_f_x_v(vmm_src, vmm_src);
            } else {
                vzext_vf4(vmm_src, vmm_tmp);
                vfcvt_f_xu_v(vmm_src, vmm_src);
            }
        }
    }

    // Convert the f32 result back to the dst dtype and store it. On entry the
    // vtype is e32 at the compute LMUL; the integer paths clamp to the dst
    // range first because vfcvt.x.f saturates only at the type bounds.
    void store_vector() {
        const data_type_t dt = data_type();
        if (dt == data_type::f32) {
            vse32_v(vmm_src, reg_dst);
        } else if (dt == data_type::f16) {
            vsetvli(x0, reg_vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
            vfncvt_f_f_w(vmm_tmp, vmm_src); // e32m2 -> e16m1
            vse16_v(vmm_tmp, reg_dst);
        } else if (dt == data_type::s32) {
            load_f32_const(freg_tmp, -2147483648.0f);
            vfmax_vf(vmm_src, vmm_src, freg_tmp);
            load_f32_const(freg_tmp, 2147483520.0f); // max_value<f32>(s32)
            vfmin_vf(vmm_src, vmm_src, freg_tmp);
            vfcvt_x_f_v(vmm_src, vmm_src);
            vse32_v(vmm_src, reg_dst);
        } else { // s8 / u8
            const bool is_s8 = dt == data_type::s8;
            load_f32_const(freg_tmp, is_s8 ? -128.0f : 0.0f);
            vfmax_vf(vmm_src, vmm_src, freg_tmp);
            load_f32_const(freg_tmp, is_s8 ? 127.0f : 255.0f);
            vfmin_vf(vmm_src, vmm_src, freg_tmp);
            if (is_s8)
                vfcvt_x_f_v(vmm_src, vmm_src);
            else
                vfcvt_xu_f_v(vmm_src, vmm_src);
            // Narrow i32(m4) -> i16(m2) -> i8(m1) through vmm_tmp; the second
            // vnsrl overlaps its source in the lowest-numbered part of the
            // group, which the narrowing overlap rule allows. Values are
            // pre-clamped so the vnsrl-by-0 truncation is exact.
            vsetvli(x0, reg_vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
            vnsrl_wi(vmm_tmp, vmm_src, 0);
            vsetvli(x0, reg_vl, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
            vnsrl_wi(vmm_tmp, vmm_tmp, 0);
            vse8_v(vmm_tmp, reg_dst);
        }
    }

    // One vector-length-agnostic pass over the `vl` elements the vsetvli in
    // load_vector() grants (the analog of x64's compute_dst(tail): vsetvli
    // subsumes the tail case).
    void compute_dst() {
        load_vector();
        eltwise_injector_->compute_vector(vmm_src.getIdx());
        if (!is_fwd_) vfmul_vv(vmm_src, vmm_src, vmm_diff_dst);
        store_vector();
    }

    void compute() {
        Label loop_start, loop_end;

        L(loop_start);
        {
            beqz(reg_work_amount, loop_end);

            compute_dst();

            // Advance pointers by vl * sizeof(dtype) using the vl granted by
            // vsetvli, not the requested work amount.
            const int dsz = dtype_size();
            if (dsz == 1) // s8 / u8
                mv(reg_bytes, reg_vl);
            else
                slli(reg_bytes, reg_vl, dsz == 4 ? 2 : 1);
            add(reg_src, reg_src, reg_bytes);
            add(reg_dst, reg_dst, reg_bytes);
            if (!is_fwd_) add(reg_diff_dst, reg_diff_dst, reg_bytes);

            sub(reg_work_amount, reg_work_amount, reg_vl);
            j_(loop_start);
        }
        L(loop_end);
    }

    void generate() override {
        // rv64 jit_generator_t has no preamble()/postamble(): the kernel is a
        // leaf function touching only caller-saved registers.
        const Reg param = a0; // first argument register (no abi_param aliases)
        ld(reg_src, param, GET_OFF(src));
        ld(reg_dst, param, GET_OFF(dst));
        if (!is_fwd_) ld(reg_diff_dst, param, GET_OFF(diff_dst));
        ld(reg_work_amount, param, GET_OFF(work_amount));

        // TODO: consider improving.
        // This piece of code is responsible for the preserve_zero function
        // being a natural restriction of this implementation. It works with any
        // dense and blocked layout, but the problem raises when blocking
        // dimension is not divisible by block size. For such case, the code
        // below should save the mask, where zero padding should be preserved
        // and apply it on register before storing into dst memory. Until
        // there's a restriction on certain blocked layouts, when this behavior
        // can be relevantly easy controlled, this will cost much from code
        // perspective and will complicate the compute logic significantly.
        compute();

        ret();
    }

private:
    void load_f32_const(const FReg &f, float val) {
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        li(reg_tmp, bits);
        fmv_w_x(f, reg_tmp);
    }

    const bool is_fwd_;

    Reg reg_src = a1;
    Reg reg_dst = a2;
    Reg reg_diff_dst = a3;
    Reg reg_work_amount = a4;
    Reg reg_vl = t0;
    Reg reg_bytes = t1;
    Reg reg_tmp = t2;

    VReg vmm_src = VReg(4);
    VReg vmm_tmp = VReg(2);
    VReg vmm_diff_dst = VReg(20);
    // Injector FP scratch fa0, dead once compute_vector() returns; reused for
    // the integer saturation bounds.
    FReg freg_tmp = fa0;

    std::unique_ptr<jit_uni_eltwise_injector_t<v>> eltwise_injector_;
};

} // namespace

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::pd_t::init(engine_t *engine) {
    const memory_desc_wrapper src_d(src_md());
    const data_type_t d_type = src_md()->data_type;

    // Runtime ISA dispatch (this primitive is pure JIT and registered via
    // CPU_INSTANCE_RV64). The zvfh instance owns f16; the v instance owns f32
    // and the integer dtypes, which reuse the f32 kernel through
    // convert-on-load / saturate-on-store.
    VDISPATCH_ELTWISE(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_ELTWISE(isa == zvfh
                    ? d_type == data_type::f16
                    : utils::one_of(d_type, data_type::f32, data_type::s32,
                              data_type::s8, data_type::u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_ELTWISE(src_md()->data_type == dst_md()->data_type,
            VERBOSE_INCONSISTENT_DT, "src", "dst");
    VDISPATCH_ELTWISE(
            platform::has_data_type_support(d_type), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "data");
    VDISPATCH_ELTWISE(src_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    // The integer dtypes keep the x64/aarch64 jit_uni_eltwise_int contract --
    // relu/linear/clip, the algs that are well defined on integer data; other
    // algs on integers -> ref.
    const bool alg_ok = utils::one_of(d_type, data_type::s32, data_type::s8,
                                data_type::u8)
            ? utils::one_of(desc_.alg_kind, alg_kind::eltwise_relu,
                      alg_kind::eltwise_linear, alg_kind::eltwise_clip)
            : eltwise_injector::is_supported(desc_.alg_kind);
    VDISPATCH_ELTWISE(alg_ok, VERBOSE_BAD_ALGORITHM);
    // refer to a comment in jit_uni_kernel why this is needed
    VDISPATCH_ELTWISE(IMPLICATION(!src_d.is_dense(), is_zero_preserved()),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_ELTWISE(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_ELTWISE(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_ELTWISE(src_d == memory_desc_wrapper(dst_md()),
            VERBOSE_INCONSISTENT_MDS, "src", "dst");

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::~jit_uni_eltwise_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto nelems = data_d.nelems(true);
    // Chunk the balanced work in cacheline units so no two threads write the
    // same destination cacheline.
    const int cacheline_elems = 64 / data_d.data_type_size();

    src += data_d.data_type_size() * data_d.offset0();
    dst += data_d.data_type_size() * data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(
                utils::div_up(nelems, cacheline_elems), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cacheline_elems);
        end = nstl::min(nelems, end * cacheline_elems);
        if (start == end) return;

        jit_args_t args;
        args.src = src + data_d.data_type_size() * start;
        args.dst = dst + data_d.data_type_size() * start;
        args.diff_dst = nullptr;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::pd_t::init(engine_t *engine) {
    // For *_use_dst_for_bwd algs the kernel reads DNNL_ARG_DST and the
    // derivative is expressed in the forward output; otherwise it reads
    // DNNL_ARG_SRC. data_md() selects the matching tensor.
    const memory_desc_wrapper data_d(data_md());
    const data_type_t d_type = data_md()->data_type;

    // Backward is float-only, matching x64/aarch64 (whose jit_uni_eltwise_bwd
    // JITs only floating dtypes and routes integers to ref_eltwise): zvfh owns
    // f16, v owns f32.
    VDISPATCH_ELTWISE(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_ELTWISE(
            isa == zvfh ? d_type == data_type::f16 : d_type == data_type::f32,
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_ELTWISE(utils::everyone_is(d_type, diff_src_md()->data_type,
                              diff_dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_ELTWISE(
            platform::has_data_type_support(d_type), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "data");
    VDISPATCH_ELTWISE(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_ELTWISE(data_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_ELTWISE(eltwise_injector::is_supported(desc_.alg_kind),
            VERBOSE_BAD_ALGORITHM);
    // refer to a comment in jit_uni_kernel why this is needed
    VDISPATCH_ELTWISE(IMPLICATION(!data_d.is_dense(), is_zero_preserved()),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    // The kernel computes f'(data) * diff_dst, walking data / diff_dst /
    // diff_src in flat lockstep, so the data tensor (src or dst) must share
    // diff_dst's layout -- otherwise a different-tag data (e.g. NCHW src vs
    // NHWC diff_dst) would be mispaired element-for-element. x64/aarch64 gate
    // data == diff_dst the same way.
    VDISPATCH_ELTWISE(data_d == memory_desc_wrapper(diff_dst_md()),
            VERBOSE_INCONSISTENT_MDS, "data", "diff_dst");
    VDISPATCH_ELTWISE(memory_desc_wrapper(diff_src_md())
                    == memory_desc_wrapper(diff_dst_md()),
            VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
    VDISPATCH_ELTWISE(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::~jit_uni_eltwise_bwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto src = pd()->use_dst() ? CTX_IN_MEM(const char *, DNNL_ARG_DST)
                               : CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());
    const auto nelems = data_d.nelems(true);
    // Chunk the balanced work in cacheline units so no two threads write the
    // same destination cacheline.
    const int cacheline_elems = 64 / data_d.data_type_size();

    src += data_d.data_type_size() * data_d.offset0();
    diff_dst += diff_data_d.data_type_size() * diff_data_d.offset0();
    diff_src += diff_data_d.data_type_size() * diff_data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(
                utils::div_up(nelems, cacheline_elems), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cacheline_elems);
        end = nstl::min(nelems, end * cacheline_elems);
        if (start == end) return;

        jit_args_t args;
        args.src = src + data_d.data_type_size() * start;
        args.dst = diff_src + diff_data_d.data_type_size() * start;
        args.diff_dst = diff_dst + diff_data_d.data_type_size() * start;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template struct jit_uni_eltwise_fwd_t<v>;
template struct jit_uni_eltwise_fwd_t<zvfh>;

template struct jit_uni_eltwise_bwd_t<v>;
template struct jit_uni_eltwise_bwd_t<zvfh>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
