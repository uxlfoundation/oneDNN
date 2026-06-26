/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/prelu/jit_uni_prelu_forward.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

namespace prelu {

static bool dims_equal(
        const dims_t &lhs_dims, const dims_t &rhs_dims, dim_t ndims) {
    for (dim_t i = 0; i < ndims; ++i)
        if (lhs_dims[i] != rhs_dims[i]) return false;
    return true;
}

static bool is_full_bcast(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d) {
    const auto src_ndims = src_d.ndims();

    // Full broadcast here means "not broadcast at all": weights match src
    // element-for-element, including the physical blocking information.
    if (src_ndims != weights_d.ndims()) return false;
    if (!dims_equal(src_d.dims(), weights_d.dims(), src_ndims)) return false;
    if (src_d.format_kind() != weights_d.format_kind()) return false;

    if (!src_d.is_blocking_desc()) return true;

    const auto &src_bd = src_d.blocking_desc();
    const auto &weights_bd = weights_d.blocking_desc();

    return src_bd.inner_nblks == weights_bd.inner_nblks
            && dims_equal(src_bd.strides, weights_bd.strides, src_d.ndims())
            && dims_equal(src_bd.inner_blks, weights_bd.inner_blks,
                    src_bd.inner_nblks)
            && dims_equal(src_bd.inner_idxs, weights_bd.inner_idxs,
                    src_bd.inner_nblks);
}

broadcasting_strategy_t get_bcast_type(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d) {
    // Preserve the existing optimized coverage while using the common
    // broadcast vocabulary. Scalar is handled by the flat generic path below;
    // per_oc variants keep their optimized paths.
    static const bcast_set_t supported_strategies {
            broadcasting_strategy_t::no_broadcast,
            broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::scalar,
    };

    const auto bcast = get_rhs_arg_broadcasting_strategy(
            *weights_d.md_, src_d, supported_strategies);

    return bcast;
}

bool bcast_supported(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, size_t simd_w) {
    const auto bcast_type = get_bcast_type(src_d, weights_d);

    if (bcast_type == broadcasting_strategy_t::scalar) return true;

    if (bcast_type == broadcasting_strategy_t::no_broadcast) {
        return is_full_bcast(src_d, weights_d);
    }

    if (bcast_type == broadcasting_strategy_t::per_oc && !src_d.is_plain()) {
        // Keeping blocked support deliberately narrow: one channel block, on the
        // channel dimension, with block size equal to the active vector width.
        const auto check_block_consistency = [=](const memory_desc_wrapper &d) {
            const auto &bd = d.blocking_desc();
            return bd.inner_nblks == 1
                    && static_cast<size_t>(bd.inner_blks[0]) == simd_w
                    && bd.inner_idxs[0] == 1;
        };

        return check_block_consistency(src_d)
                && check_block_consistency(weights_d);
    }

    if (!utils::one_of(bcast_type, broadcasting_strategy_t::per_oc,
                broadcasting_strategy_t::per_oc_spatial))
        return false;

    const auto &src_strides = src_d.blocking_desc().strides;
    const auto &weights_strides = weights_d.blocking_desc().strides;

    return src_strides[0] >= src_strides[1]
            && IMPLICATION(src_strides[1] > 1, src_strides[1] >= src_strides[2])
            && weights_strides[0] >= weights_strides[1];
}

} // namespace prelu

class jit_prelu_forward_kernel_t : public jit_generator_t {
public:
    // Per-call kernel state. The C++ primitive wrapper slices the tensor into
    // contiguous chunks and passes each chunk through this compact ABI.
    struct call_params_t {
        const void *src = nullptr;
        const void *weights = nullptr;
        void *dst = nullptr;
        size_t compute_data_size = 0;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_prelu_forward_kernel_t)

    static jit_prelu_forward_kernel_t *create(cpu_isa_t isa,
            broadcasting_strategy_t bcast, bool per_oc_blocked,
            data_type_t data_type);

    ~jit_prelu_forward_kernel_t() override = default;

    void operator()(call_params_t *params) {
        jit_generator_t::operator()(params);
    }

    virtual size_t simd_w() const noexcept = 0;
    virtual broadcasting_strategy_t get_bcast() const noexcept = 0;
    virtual bool per_oc_blocked() const noexcept = 0;

protected:
    explicit jit_prelu_forward_kernel_t(
            broadcasting_strategy_t bcast, bool per_oc_blocked)
        : bcast_(bcast), per_oc_blocked_(per_oc_blocked) {}

    const broadcasting_strategy_t bcast_;
    const bool per_oc_blocked_;
};

template <cpu_isa_t isa>
class jit_uni_prelu_forward_kernel_t : public jit_prelu_forward_kernel_t {
public:
    explicit jit_uni_prelu_forward_kernel_t(broadcasting_strategy_t bcast,
            bool per_oc_blocked, data_type_t data_type);

    size_t simd_w() const noexcept override { return simd_w_; }
    broadcasting_strategy_t get_bcast() const noexcept override {
        return bcast_;
    }
    bool per_oc_blocked() const noexcept override { return per_oc_blocked_; }

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    // Code-generation steps for the emitted function body.
    void generate() override;
    void load_params();
    void prepare_const_registers();
    void vector_loop();
    void scalar_loop();

    // ISA-specific helpers. The overloads let the common kernel body emit
    // either ASIMD or SVE instructions from the same template.
    void load_vector(const VReg4S &dst, const XReg &addr);
    void load_vector(const ZRegS &dst, const XReg &addr);
    void store_vector(const XReg &addr, const VReg4S &src);
    void store_vector(const XReg &addr, const ZRegS &src);
    void broadcast_weight(const VReg4S &dst, const XReg &addr);
    void broadcast_weight(const ZRegS &dst, const XReg &addr);
    void compute_vector(const VReg4S &src, const VReg4S &weights);
    void compute_vector(const ZRegS &src, const ZRegS &weights);
    void load_data_vector(const VReg4S &dst, const XReg &addr);
    void load_data_vector(const ZRegS &dst, const XReg &addr);
    void store_data_vector(const XReg &addr, const VReg4S &src);
    void store_data_vector(const XReg &addr, const ZRegS &src);

    bool weights_are_const() const noexcept;
    bool weights_are_vector_const() const noexcept;
    bool weights_are_scalar_const() const noexcept;
    size_t data_type_size() const noexcept;

    const data_type_t data_type_;
    const size_t simd_w_;
    const size_t simd_bytes_;

    // General-purpose registers used by the generated kernel.
    const XReg reg_src_ = x8;
    const XReg reg_weights_ = x9;
    const XReg reg_dst_ = x10;
    const XReg reg_work_ = x11;

    // Vector/SVE registers. TRegS maps to VReg4S for ASIMD and ZRegS for SVE.
    const TRegS v_src_ = TRegS(0);
    const TRegS v_min_ = TRegS(2);
    const TRegS v_weights_ = TRegS(3);
    const TRegS v_zero_ = TRegS(4);

    // Scalar registers used for the cleanup loop after full vectors are done.
    const SReg s_src_ = s0;
    const SReg s_max_ = s1;
    const SReg s_min_ = s2;
    const SReg s_weights_ = s3;
    const SReg s_zero_ = s4;
};

#define PARAM_OFF(x) offsetof(jit_prelu_forward_kernel_t::call_params_t, x)

template <cpu_isa_t isa>
jit_uni_prelu_forward_kernel_t<isa>::jit_uni_prelu_forward_kernel_t(
        broadcasting_strategy_t bcast, bool per_oc_blocked,
        data_type_t data_type)
    : jit_prelu_forward_kernel_t(bcast, per_oc_blocked)
    , data_type_(data_type)
    , simd_w_(simd_elems(data_type::f32, isa))
    , simd_bytes_(simd_w_ * types::data_type_size(data_type_)) {}

template <cpu_isa_t isa>
bool jit_uni_prelu_forward_kernel_t<isa>::weights_are_vector_const()
        const noexcept {
    return per_oc_blocked_;
}

template <cpu_isa_t isa>
bool jit_uni_prelu_forward_kernel_t<isa>::weights_are_scalar_const()
        const noexcept {
    return utils::one_of(bcast_, broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::scalar);
}

template <cpu_isa_t isa>
bool jit_uni_prelu_forward_kernel_t<isa>::weights_are_const() const noexcept {
    return weights_are_vector_const() || weights_are_scalar_const();
}

template <cpu_isa_t isa>
size_t jit_uni_prelu_forward_kernel_t<isa>::data_type_size() const noexcept {
    return types::data_type_size(data_type_);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_vector(
        const VReg4S &dst, const XReg &addr) {
    ld1(dst, ptr(addr));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_vector(
        const ZRegS &dst, const XReg &addr) {
    ld1w(dst, P_ALL_ONE / T_z, ptr(addr));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_data_vector(
        const VReg4S &dst, const XReg &addr) {
    if (data_type_ == data_type::f32) {
        ld1(dst, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        ld1(VReg4H(dst.getIdx()), ptr(addr));
        fcvtl(dst, VReg4H(dst.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_data_vector(
        const ZRegS &dst, const XReg &addr) {
    if (data_type_ == data_type::f32) {
        ld1w(dst, P_ALL_ONE / T_z, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        ld1h(dst, P_ALL_ONE / T_z, ptr(addr));
        fcvt(dst, P_ALL_ONE / T_m, ZRegH(dst.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::store_vector(
        const XReg &addr, const VReg4S &src) {
    st1(src, ptr(addr));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::store_vector(
        const XReg &addr, const ZRegS &src) {
    st1w(src, P_ALL_ONE / T_z, ptr(addr));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::store_data_vector(
        const XReg &addr, const VReg4S &src) {
    if (data_type_ == data_type::f32) {
        st1(src, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        fcvtn(VReg4H(src.getIdx()), src);
        st1(VReg4H(src.getIdx()), ptr(addr));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::store_data_vector(
        const XReg &addr, const ZRegS &src) {
    if (data_type_ == data_type::f32) {
        st1w(src, P_ALL_ONE / T_z, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        fcvt(ZRegH(src.getIdx()), P_ALL_ONE / T_m, src);
        st1h(src, P_ALL_ONE, ptr(addr));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::broadcast_weight(
        const VReg4S &dst, const XReg &addr) {
    if (data_type_ == data_type::f32) {
        ld1r(dst, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        ld1r(VReg4H(dst.getIdx()), ptr(addr));
        fcvtl(dst, VReg4H(dst.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::broadcast_weight(
        const ZRegS &dst, const XReg &addr) {
    if (data_type_ == data_type::f32) {
        ld1rw(dst, P_ALL_ONE / T_z, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        ld1rh(dst, P_ALL_ONE / T_z, ptr(addr));
        fcvt(dst, P_ALL_ONE / T_m, ZRegH(dst.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::compute_vector(
        const VReg4S &src, const VReg4S &weights) {
    const VReg4S v_min(v_min_.getIdx());
    const VReg4S v_zero(v_zero_.getIdx());

    // PReLU: positive part passes through, negative part is multiplied by
    // weights. Algebraically: max(src, 0) + weights * min(src, 0).
    fmin(v_min, src, v_zero);
    fmax(src, src, v_zero);
    fmla(src, v_min, weights);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::compute_vector(
        const ZRegS &src, const ZRegS &weights) {
    const ZRegS v_min(v_min_.getIdx());

    // Same math as the ASIMD version. The SVE immediate fmin/fmax forms are
    // destructive, so keep one copy for the negative part and compute the
    // positive part in src directly.
    mov(v_min, P_ALL_ONE / T_m, src);
    fmin(v_min, P_ALL_ONE / T_m, 0.0f);
    fmax(src, P_ALL_ONE / T_m, 0.0f);
    fmla(src, P_ALL_ONE / T_m, v_min, weights);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_params() {
    ldr(reg_src_, ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(src))));
    ldr(reg_weights_,
            ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(weights))));
    ldr(reg_dst_, ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(dst))));
    ldr(reg_work_,
            ptr(abi_param1,
                    static_cast<uint32_t>(PARAM_OFF(compute_data_size))));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::prepare_const_registers() {
    uni_clear(TReg(v_zero_.getIdx()));

    // Some broadcast modes can load weights once per kernel call. For full and
    // NHWC-like cases, weights advance with src and are loaded in vector_loop().
    if (weights_are_vector_const())
        load_data_vector(v_weights_, reg_weights_);
    else if (weights_are_scalar_const())
        broadcast_weight(v_weights_, reg_weights_);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::vector_loop() {

    asm_for_step(reg_work_, reg_work_, simd_w_, [&]() {
        load_data_vector(v_src_, reg_src_);
        // weights are either constant or loaded in the loop, depending on the
        // broadcast mode.
        if (!weights_are_const()) load_data_vector(v_weights_, reg_weights_);
        compute_vector(v_src_, v_weights_);
        store_data_vector(reg_dst_, v_src_);

        add_imm(reg_src_, reg_src_, simd_bytes_, X_TMP_0);
        add_imm(reg_dst_, reg_dst_, simd_bytes_, X_TMP_0);
        if (!weights_are_const())
            add_imm(reg_weights_, reg_weights_, simd_bytes_, X_TMP_0);
    });

    // The vector loop may leave a small number of elements unprocessed. The
    // scalar loop handles the remainder. The vector loop is guaranteed to
    // terminate with reg_work_ < simd_w_.
    Label tail_ready;
    cmp(reg_work_, 0);
    bge(tail_ready);
    add_imm(reg_work_, reg_work_, simd_w_, X_TMP_0);
    L(tail_ready);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::scalar_loop() {
    asm_for(reg_work_, reg_work_, [&]() {
        if (data_type_ == data_type::f32) {
            ldr(s_src_, ptr(reg_src_));
            ldr(s_weights_, ptr(reg_weights_));
        } else if (data_type_ == data_type::f16) {
            ldr(HReg(s_src_.getIdx()), ptr(reg_src_));
            ldr(HReg(s_weights_.getIdx()), ptr(reg_weights_));
            fcvt(s_src_, HReg(s_src_.getIdx()));
            fcvt(s_weights_, HReg(s_weights_.getIdx()));
        } else {
            assert(!"unsupported PReLU data type");
        }
        // Scalar cleanup mirrors the vector formula for plain-layout tails.
        fmax(s_max_, s_src_, s_zero_);
        fmin(s_min_, s_src_, s_zero_);
        fmul(s_min_, s_min_, s_weights_);
        fadd(s_max_, s_max_, s_min_);
        if (data_type_ == data_type::f32) {
            str(s_max_, ptr(reg_dst_));
        } else if (data_type_ == data_type::f16) {
            fcvt(HReg(s_max_.getIdx()), s_max_);
            str(HReg(s_max_.getIdx()), ptr(reg_dst_));
        }

        add_imm(reg_src_, reg_src_, data_type_size(), X_TMP_0);
        add_imm(reg_dst_, reg_dst_, data_type_size(), X_TMP_0);
        if (!weights_are_const())
            add_imm(reg_weights_, reg_weights_, data_type_size(), X_TMP_0);
    });
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::generate() {
    preamble();
    load_params();
    prepare_const_registers();
    vector_loop();
    scalar_loop();
    postamble();
}

#undef PARAM_OFF

jit_prelu_forward_kernel_t *jit_prelu_forward_kernel_t::create(cpu_isa_t isa,
        broadcasting_strategy_t bcast, bool per_oc_blocked,
        data_type_t data_type) {
    if (isa == sve) {
        return new jit_uni_prelu_forward_kernel_t<sve>(
                bcast, per_oc_blocked, data_type);
    } else if (isa == asimd) {
        return new jit_uni_prelu_forward_kernel_t<asimd>(
                bcast, per_oc_blocked, data_type);
    }

    assert(!"unsupported PReLU JIT ISA");
    return nullptr;
}

template <cpu_isa_t isa>
status_t jit_uni_prelu_fwd_t<isa>::pd_t::init(engine_t *engine) {
    UNUSED(engine);

    const memory_desc_wrapper src_d {src_md(0)};
    const memory_desc_wrapper weights_d {weights_md(0)};
    const memory_desc_wrapper dst_d {dst_md(0)};
    const auto simd_w = simd_elems(data_type::f32, isa);

    VDISPATCH_PRELU(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_PRELU(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_PRELU(src_d.data_type() == dst_d.data_type(),
            VERBOSE_INCONSISTENT_DT, "src", "dst");
    VDISPATCH_PRELU(utils::everyone_is(src_d.data_type(), weights_d.data_type(),
                            dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_PRELU(
            utils::one_of(src_d.data_type(), data_type::f32, data_type::f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_PRELU(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
    VDISPATCH_PRELU(src_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_PRELU(weights_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_PRELU(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_PRELU(dst_d == src_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
    VDISPATCH_PRELU(prelu::bcast_supported(src_d, weights_d, simd_w),
            VERBOSE_UNSUPPORTED_DT_CFG);

    bcast_ = prelu::get_bcast_type(src_d, weights_d);
    per_oc_blocked_
            = bcast_ == broadcasting_strategy_t::per_oc && !src_d.is_plain();
    data_type_ = src_d.data_type();
    return status::success;
}

template <cpu_isa_t isa>
jit_uni_prelu_fwd_t<isa>::jit_uni_prelu_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_prelu_fwd_t<isa>::~jit_uni_prelu_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_prelu_fwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);

    CHECK(safe_ptr_assign(kernel_,
            jit_prelu_forward_kernel_t::create(isa, pd()->bcast_,
                    pd()->per_oc_blocked_, pd()->data_type_)));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_prelu_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    using byte = unsigned char;

    const byte *const src = CTX_IN_MEM(const byte *, DNNL_ARG_SRC);
    const byte *const weights = CTX_IN_MEM(const byte *, DNNL_ARG_WEIGHTS);
    byte *const dst = CTX_OUT_MEM(byte *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d {pd()->src_md(0)};
    const auto kernel = kernel_.get();
    const auto bcast = kernel->get_bcast();
    const size_t simd_w = kernel->simd_w();
    const size_t data_type_size = types::data_type_size(pd()->data_type_);

    const dim_t ndims = src_d.ndims();

    if (utils::one_of(bcast, broadcasting_strategy_t::no_broadcast,
                broadcasting_strategy_t::scalar)) {
        const bool is_scalar = bcast == broadcasting_strategy_t::scalar;
        const dim_t nelems = src_d.nelems(true);
        const dim_t work_chunks = utils::div_up(nelems, (dim_t)simd_w);

        // Full weights are laid out exactly like src/dst. Scalar weights use
        // the same flat work split but always point at weights[0].
        parallel(0, [=](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(work_chunks, nthr, ithr, start, end);
            start = nstl::min(nelems, start * (dim_t)simd_w);
            end = nstl::min(nelems, end * (dim_t)simd_w);
            if (start >= end) return;

            jit_prelu_forward_kernel_t::call_params_t params;
            params.compute_data_size = end - start;
            params.src = src + start * data_type_size;
            params.weights
                    = is_scalar ? weights : weights + start * data_type_size;
            params.dst = dst + start * data_type_size;
            (*kernel)(&params);
        });

        return status::success;
    }

    const dim_t MB = pd()->N();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const dim_t SP = D * H * W;

    const dim_t nelems_single_mb
            = utils::array_product(src_d.padded_dims() + 1, ndims - 1);

    if (bcast == broadcasting_strategy_t::per_oc && !kernel->per_oc_blocked()) {
        // NHWC-like case. For each minibatch/spatial position, process the
        // contiguous channel dimension and advance weights alongside channels.
        parallel_nd(MB, SP, [=](dim_t mb, dim_t sp) {
            const dim_t offset = mb * nelems_single_mb + sp * C;

            jit_prelu_forward_kernel_t::call_params_t params;
            params.compute_data_size = C;
            params.src = src + offset * data_type_size;
            params.weights = weights;
            params.dst = dst + offset * data_type_size;
            (*kernel)(&params);
        });
    } else if (bcast == broadcasting_strategy_t::per_oc_spatial) {
        // NCHW-like case. Each kernel call handles one channel's spatial range,
        // so a single scalar weight can be broadcast and reused.
        parallel_nd(MB, C, [=](dim_t mb, dim_t c) {
            const dim_t offset = mb * nelems_single_mb + c * SP;

            jit_prelu_forward_kernel_t::call_params_t params;
            params.compute_data_size = SP;
            params.src = src + offset * data_type_size;
            params.weights = weights + c * data_type_size;
            params.dst = dst + offset * data_type_size;
            (*kernel)(&params);
        });
    } else if (bcast == broadcasting_strategy_t::per_oc
            && kernel->per_oc_blocked()) {
        const dim_t C_blocks = utils::div_up(C, (dim_t)simd_w);

        // Blocked channel layout. One vector of channel weights is loaded for a
        // block and reused across every spatial element in that block.
        parallel_nd(MB, C_blocks, [=](dim_t mb, dim_t c_blk) {
            const dim_t offset = mb * nelems_single_mb + c_blk * SP * simd_w;

            jit_prelu_forward_kernel_t::call_params_t params;
            params.compute_data_size = SP * simd_w;
            params.src = src + offset * data_type_size;
            params.weights = weights + c_blk * simd_w * data_type_size;
            params.dst = dst + offset * data_type_size;
            (*kernel)(&params);
        });
    } else {
        assert(!"unsupported PReLU broadcast type");
        return status::runtime_error;
    }

    return status::success;
}

template class jit_uni_prelu_forward_kernel_t<asimd>;
template class jit_uni_prelu_forward_kernel_t<sve>;
template struct jit_uni_prelu_fwd_t<asimd>;
template struct jit_uni_prelu_fwd_t<sve>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
