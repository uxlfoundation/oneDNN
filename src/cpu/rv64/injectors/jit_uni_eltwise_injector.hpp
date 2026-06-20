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
#ifndef CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP
#define CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/rv64/injectors/injector_utils.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace eltwise_injector {

// Caller-provided scratch the eltwise injector may freely clobber inside
// compute_vector(). The host kernel guarantees these registers are dead during
// post-op application. RVV vector length is a run-time quantity, so the host
// (which owns the vsetvli state and the accumulator allocation) supplies free
// vector scratch rather than the injector spilling. The injector may use v0 as
// a mask register, so hosts must keep accumulator groups away from v0.
struct static_params_t {
    static_params_t(const Xbyak_riscv::VReg &v_aux0,
            const Xbyak_riscv::VReg &v_aux1, const Xbyak_riscv::VReg &v_aux2,
            const Xbyak_riscv::FReg &f_aux0, const Xbyak_riscv::FReg &f_aux1,
            const Xbyak_riscv::Reg &gpr_aux0, bool is_fwd = true)
        : v_aux0(v_aux0)
        , v_aux1(v_aux1)
        , v_aux2(v_aux2)
        , f_aux0(f_aux0)
        , f_aux1(f_aux1)
        , gpr_aux0(gpr_aux0)
        , is_fwd(is_fwd) {}

    // Up to three vector scratch groups (same LMUL as the host accumulator).
    // Forward arithmetic algorithms use at most v_aux0; exp/logistic use
    // v_aux0 and v_aux2; tanh/elu/swish/gelu_tanh use all three. Forward and
    // backward may use v0 as a mask.
    Xbyak_riscv::VReg v_aux0, v_aux1, v_aux2;
    Xbyak_riscv::FReg f_aux0, f_aux1; // two FP scratch regs for constants
    Xbyak_riscv::Reg gpr_aux0; // one GPR scratch for constant materialization
    // Forward (d = alg(s)) or backward (ds = alg'(s)).
    bool is_fwd;
};

// Whether the JIT eltwise injector can emit this algorithm. Covers the
// mask-free arithmetic forward algorithms plus the transcendentals built on the
// inline-coefficient exp() primitive (exp/logistic/tanh/elu/swish/gelu_tanh).
// Algorithms still needing log/erf (mish, gelu_erf, soft_relu, pow) are not yet
// supported and fall back to a reference impl (the consumer pd rejects them).
bool is_alg_supported(alg_kind_t alg);

} // namespace eltwise_injector

// In-kernel forward eltwise post-op injector for RVV.
//
// compute_vector(v) applies `scale * alg(alpha, beta, v)` element-wise to the
// active lanes of vector register group `v`, in place. The host sets the vtype
// (SEW=e32 / data LMUL) and the active vl via vsetvli before calling; the
// injector emits only data-path instructions and never changes vl/vtype.
template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_t {
    using Vmm = typename jit_isa_traits_t<isa>::Vmm;

    jit_uni_eltwise_injector_t(jit_generator_t *host, alg_kind_t alg,
            float alpha, float beta, float scale,
            const eltwise_injector::static_params_t &sp)
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , h_(host)
        , is_fwd_(sp.is_fwd)
        , v_aux0_(sp.v_aux0)
        , v_aux1_(sp.v_aux1)
        , v_aux2_(sp.v_aux2)
        , f_aux0_(sp.f_aux0)
        , f_aux1_(sp.f_aux1)
        , gpr_aux0_(sp.gpr_aux0) {
        assert(eltwise_injector::is_alg_supported(alg_));
    }

    jit_uni_eltwise_injector_t(jit_generator_t *host,
            const post_ops_t::entry_t::eltwise_t &e,
            const eltwise_injector::static_params_t &sp)
        : jit_uni_eltwise_injector_t(
                  host, e.alg, e.alpha, e.beta, e.scale, sp) {}

    // ARM/x64-style index-based interface: apply the eltwise op to the vector
    // register group(s) identified by index. Forward: d = scale * alg(s).
    // Backward (is_fwd=false): transforms `s` into the derivative alg'(s); the
    // caller multiplies by diff_dst to get diff_src.
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }
    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs);

private:
    // Worker: applies the op to one register group, dispatching on alg/is_fwd.
    void compute_body(const Vmm &v);
    void compute_vector_bwd(const Vmm &v);
    void load_f32_const(const Xbyak_riscv::FReg &f, float val);
    // NaN-preserving clamp(v, lo, hi); reuses f_aux0_/f_aux1_ and v0.
    void clamp(const Vmm &v, float lo, float hi);
    // exp(v) in place: range-reduce + Horner poly + 2^n scale. Uses v_aux0
    // and v_aux2; leaves v_aux1 free for callers. Building block for the
    // other transcendentals.
    void exp_compute_vector(const Vmm &v);
    // sigmoid(v) = 1 / (1 + exp(-v)), in place. Uses the exp building block.
    void logistic_compute_vector(const Vmm &v);

    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;

    jit_generator_t *const h_;
    const bool is_fwd_;
    const Xbyak_riscv::VReg v_aux0_;
    const Xbyak_riscv::VReg v_aux1_;
    const Xbyak_riscv::VReg v_aux2_;
    const Xbyak_riscv::FReg f_aux0_;
    const Xbyak_riscv::FReg f_aux1_;
    const Xbyak_riscv::Reg gpr_aux0_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP
