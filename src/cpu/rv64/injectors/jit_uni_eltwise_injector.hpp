/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#ifndef CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP
#define CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/injectors/injector_utils.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace eltwise_injector {

// Caller-provided scratch the eltwise injector may freely clobber inside
// compute_vector(). The host kernel guarantees these registers are dead during
// post-op application. RVV vector length is a run-time quantity and an aux is
// a whole LMUL-aligned register group, so the host (which owns the vsetvli
// state and the accumulator layout) supplies free vector scratch rather than
// the injector spilling to stack as on x64. The injector may use v0 as a mask
// register, so hosts must keep accumulator groups away from v0.
struct static_params_t {
    static_params_t(const Xbyak_riscv::VReg &v_aux0,
            const Xbyak_riscv::VReg &v_aux1, const Xbyak_riscv::VReg &v_aux2,
            const Xbyak_riscv::VReg &v_aux3, const Xbyak_riscv::VReg &v_aux4,
            const Xbyak_riscv::FReg &f_aux0, const Xbyak_riscv::FReg &f_aux1,
            const Xbyak_riscv::Reg &gpr_aux0, bool is_fwd)
        : v_aux0(v_aux0)
        , v_aux1(v_aux1)
        , v_aux2(v_aux2)
        , v_aux3(v_aux3)
        , v_aux4(v_aux4)
        , f_aux0(f_aux0)
        , f_aux1(f_aux1)
        , gpr_aux0(gpr_aux0)
        , is_fwd(is_fwd) {}

    // Up to five vector scratch groups (same LMUL as the host accumulator).
    // Forward arithmetic algorithms use at most v_aux0; exp/logistic use
    // v_aux0 and v_aux2; tanh/elu/swish/gelu_tanh/mish use all three;
    // soft_relu/log/gelu_erf (forward) additionally use v_aux3, and some
    // backward algorithms use both v_aux3 and v_aux4 (see aux_vecs_count()).
    // A host that only enables algorithms within a smaller budget may pass
    // v_aux4 = v_aux3 (all current 4-group post-op consumers do).
    // Forward and backward may use v0 as a mask.
    Xbyak_riscv::VReg v_aux0, v_aux1, v_aux2, v_aux3, v_aux4;
    Xbyak_riscv::FReg f_aux0, f_aux1; // two FP scratch regs for constants
    Xbyak_riscv::Reg gpr_aux0; // one GPR scratch for constant materialization
    // Forward (d = alg(s)) or backward (ds = alg'(s)).
    bool is_fwd;
};

/*
 * Checks if a forward eltwise algorithm is supported by the eltwise injector
 * within the base 3-aux scratch budget every post-op consumer provides.
 * eltwise_pow is not supported (like aarch64) and falls back to a reference
 * implementation.
 */
bool is_alg_supported(alg_kind_t alg);

/*
 * Checks if a forward algorithm needs a 4th vector aux (v_aux3) on top of the
 * base budget. A post-op consumer may enable such an algorithm only when it
 * supplies that extra scratch (see the post_ops_ok n_vaux argument).
 */
bool needs_extra_aux(alg_kind_t alg);

/*
 * Checks if the eltwise injector supports the algorithm: the is_alg_supported()
 * base set plus the algorithms needing the extra aux. A single set drives both
 * directions (as on x64/aarch64); eltwise_round is forward-only (no
 * derivative), but the common layer rejects round + backward before this runs,
 * so it is not special-cased. Unlike x64/aarch64 there is no isa or dt
 * argument: rv64 has a single runtime-gated vector ISA (the primitive
 * descriptor checks mayiuse()) and the dt is validated in the descriptor.
 */
bool is_supported(alg_kind_t alg);

} // namespace eltwise_injector

// In-kernel eltwise (post-op) injector for RVV.
//
// compute_vector(v) applies `scale * alg(alpha, beta, v)` element-wise to the
// active lanes of vector register group `v`, in place (backward: transforms
// `v` into the derivative alg'(v); the caller multiplies by diff_dst). The
// host sets the vtype (SEW=e32 / data LMUL) and the active vl via vsetvli
// before calling; the injector emits only data-path instructions and never
// changes vl/vtype.
template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_t {
    using Vmm = typename jit_isa_traits_t<isa>::Vmm;

    // Arguments description:
    // host - jit generator which is filled with instructions
    // alg, alpha, beta, scale - user eltwise arguments
    // sp - host-provided scratch registers and direction (see static_params_t;
    //   rv64 has no save_state/p_table/k_mask: there is no constants table --
    //   RVV vf-form instructions take FP scalars directly -- no stack
    //   spilling, and the mask register is architecturally v0).
    // use_dst is derived from alg: backward descriptors carry the
    //   *_use_dst_for_bwd kinds (x64 receives it as a constructor argument).
    jit_uni_eltwise_injector_t(jit_generator_t *host, alg_kind_t alg,
            float alpha, float beta, float scale,
            const eltwise_injector::static_params_t &sp)
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , h_(host)
        , is_fwd_(sp.is_fwd)
        , use_dst_(!sp.is_fwd
                  && utils::one_of(alg, alg_kind::eltwise_relu_use_dst_for_bwd,
                          alg_kind::eltwise_tanh_use_dst_for_bwd,
                          alg_kind::eltwise_elu_use_dst_for_bwd,
                          alg_kind::eltwise_sqrt_use_dst_for_bwd,
                          alg_kind::eltwise_logistic_use_dst_for_bwd,
                          alg_kind::eltwise_exp_use_dst_for_bwd,
                          alg_kind::eltwise_clip_v2_use_dst_for_bwd))
        , v_aux0_(sp.v_aux0)
        , v_aux1_(sp.v_aux1)
        , v_aux2_(sp.v_aux2)
        , v_aux3_(sp.v_aux3)
        , v_aux4_(sp.v_aux4)
        , f_aux0_(sp.f_aux0)
        , f_aux1_(sp.f_aux1)
        , gpr_aux0_(sp.gpr_aux0) {
        assert(eltwise_injector::is_supported(alg_));
        // An algorithm reaching into the 4th/5th group needs distinct
        // registers there; hosts without the budget alias v_aux4 = v_aux3 and
        // must gate such algorithms out (see needs_extra_aux()).
        assert(IMPLICATION(aux_vecs_count(alg_, is_fwd_) >= 4,
                !utils::one_of(v_aux3_.getIdx(), v_aux0_.getIdx(),
                        v_aux1_.getIdx(), v_aux2_.getIdx())));
        assert(IMPLICATION(aux_vecs_count(alg_, is_fwd_) >= 5,
                !utils::one_of(v_aux4_.getIdx(), v_aux0_.getIdx(),
                        v_aux1_.getIdx(), v_aux2_.getIdx(), v_aux3_.getIdx())));
    }

    jit_uni_eltwise_injector_t(jit_generator_t *host,
            const post_ops_t::entry_t::eltwise_t &e,
            const eltwise_injector::static_params_t &sp)
        : jit_uni_eltwise_injector_t(
                  host, e.alg, e.alpha, e.beta, e.scale, sp) {}

    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs);
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }

    // This call is `static` and `public` so a host can size its
    // static_params_t before constructing the injector. Unlike x64 (which
    // counts exact spill slots), rv64 counts required aux vector *groups* at
    // the applicability-contract granularity: 3 for the base post-op budget,
    // 4 for the forward algorithms needing v_aux3, 5 for backward
    // (standalone-only).
    static size_t aux_vecs_count(alg_kind_t alg, bool is_fwd);

private:
    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;

    jit_generator_t *const h_;

    const bool is_fwd_;
    const bool use_dst_;
    const Xbyak_riscv::VReg v_aux0_;
    const Xbyak_riscv::VReg v_aux1_;
    const Xbyak_riscv::VReg v_aux2_;
    const Xbyak_riscv::VReg v_aux3_;
    const Xbyak_riscv::VReg v_aux4_;
    const Xbyak_riscv::FReg f_aux0_;
    const Xbyak_riscv::FReg f_aux1_;
    const Xbyak_riscv::Reg gpr_aux0_;
    // The RVV mask register is architecturally fixed: vmerge/vfmerge read
    // v0.t, so unlike x64's assignable k_mask this is not a parameter.
    const Xbyak_riscv::VReg vmm_mask_ = Xbyak_riscv::VReg(0);

    // Worker: applies the op to one register group, dispatching on
    // alg_/is_fwd_.
    void compute_body(const Vmm &vmm_src);

    void load_f32_const(const Xbyak_riscv::FReg &f, float val);
    // NaN-preserving clamp(v, lo, hi); reuses f_aux0_/f_aux1_ and v0.
    void clamp(const Vmm &vmm_src, float lo, float hi);
    // Building blocks without an x64 analog (x64 keeps every algorithm
    // self-contained on its constants table):
    // - log_compute_vector: raw Cephes logf (frexp + degree-8 mantissa poly),
    //   no domain-edge patching; shared by log_compute_vector_fwd (which adds
    //   the 0/inf/negative patches) and soft_relu (which feeds 1+exp(x) > 0).
    //   Uses v_aux0 and v_aux2 plus v0 as a mask; leaves v_aux1 free.
    // - erf_compute_vector: sign-free Abramowitz-Stegun 7.1.26 erf(|x|)
    //   (t-poly * exp(-x^2)); shared by gelu_erf forward and backward. Uses
    //   v_aux0/v_aux1/v_aux2 and v0; leaves v_aux3/v_aux4 untouched for any
    //   values the caller must preserve.
    void log_compute_vector(const Vmm &vmm_src);
    void erf_compute_vector(const Vmm &vmm_src);

    void exp_compute_vector_fwd(const Vmm &vmm_src);
    void relu_compute_vector_fwd(const Vmm &vmm_src);
    void relu_zero_ns_compute_vector_fwd(const Vmm &vmm_src);
    void elu_compute_vector_fwd(const Vmm &vmm_src);
    void tanh_compute_vector_fwd(const Vmm &vmm_src);
    void square_compute_vector_fwd(const Vmm &vmm_src);
    void abs_compute_vector_fwd(const Vmm &vmm_src);
    void sqrt_compute_vector_fwd(const Vmm &vmm_src);
    void linear_compute_vector_fwd(const Vmm &vmm_src);
    void soft_relu_compute_vector_fwd(const Vmm &vmm_src);
    void mish_compute_vector_fwd(const Vmm &vmm_src);
    void logistic_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_tanh_compute_vector_fwd(const Vmm &vmm_src);
    void swish_compute_vector_fwd(const Vmm &vmm_src);
    void log_compute_vector_fwd(const Vmm &vmm_src);
    void clip_compute_vector_fwd(const Vmm &vmm_src);
    void clip_v2_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_erf_compute_vector_fwd(const Vmm &vmm_src);
    void round_compute_vector_fwd(const Vmm &vmm_src);
    void hardswish_compute_vector_fwd(const Vmm &vmm_src);
    void hardsigmoid_compute_vector_fwd(const Vmm &vmm_src);

    void exp_compute_vector_bwd(const Vmm &vmm_src);
    void relu_compute_vector_bwd(const Vmm &vmm_src);
    void elu_compute_vector_bwd(const Vmm &vmm_src);
    void tanh_compute_vector_bwd(const Vmm &vmm_src);
    void square_compute_vector_bwd(const Vmm &vmm_src);
    void abs_compute_vector_bwd(const Vmm &vmm_src);
    void sqrt_compute_vector_bwd(const Vmm &vmm_src);
    void linear_compute_vector_bwd(const Vmm &vmm_src);
    void soft_relu_compute_vector_bwd(const Vmm &vmm_src);
    void logistic_compute_vector_bwd(const Vmm &vmm_src);
    void mish_compute_vector_bwd(const Vmm &vmm_src);
    void gelu_tanh_compute_vector_bwd(const Vmm &vmm_src);
    void swish_compute_vector_bwd(const Vmm &vmm_src);
    void log_compute_vector_bwd(const Vmm &vmm_src);
    void clip_compute_vector_bwd(const Vmm &vmm_src);
    void clip_v2_compute_vector_bwd(const Vmm &vmm_src);
    void gelu_erf_compute_vector_bwd(const Vmm &vmm_src);
    void hardswish_compute_vector_bwd(const Vmm &vmm_src);
    void hardsigmoid_compute_vector_bwd(const Vmm &vmm_src);
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP
