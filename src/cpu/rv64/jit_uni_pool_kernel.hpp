/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_JIT_UNI_POOL_KERNEL_HPP
#define CPU_RV64_JIT_UNI_POOL_KERNEL_HPP

#include <functional>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// x64/aarch64-style pooling kernel for the blocked (nChw{c_block}c) and nspc
// layouts. Faithful port of aarch64 jit_uni_pool_kernel_t: the pooling window is
// baked into the generated code, ur output columns share loaded inputs, and
// channels are the vector dimension (one m1 register == c_block f32 lanes,
// c_block = min(VLEN/32, 16), one of {4,8,16}). Covers forward inference,
// forward training (max index workspace), and backward (max via index, avg).
// f16 accumulates in f32 (widen on load, narrow on store, requires zvfh).
//
// RVV adaptation notes:
//   - ur_bc is always 1 (one channel block per call); the driver loops nb_c
//     blocks. Channel tail (nspc, C not a multiple of c_block) is handled by
//     vsetvli(vl=c_tail) for the last block — no predicate masks.
//   - blocked with padded channels (is_c_padded) is declined to the reference.
//   - Register map: v0=mask, v1..v3=eltwise injector scratch, v4=binary rhs
//     scratch, v5..v7=compute transients, v8..v31=24-register acc/input/index
//     tile (vreg(i)=v8+i). Post-ops apply after the window loop (inputs dead).
template <cpu_isa_t isa>
struct jit_uni_pool_kernel_t : public jit_generator_t {

    jit_uni_pool_kernel_t(
            const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md);
    ~jit_uni_pool_kernel_t() override;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_kernel_t)

    static status_t init_conf(jit_pool_conf_t &jpp,
            memory_tracking::registrar_t &scratchpad, primitive_attr_t &attr,
            const pooling_pd_t *ppd);

    void operator()(const jit_uni_pooling_args_t *p) const {
        jit_generator_t::operator()(p);
    }

    jit_pool_conf_t jpp;

private:
    using Vmm = Xbyak_riscv::VReg;
    using Reg = Xbyak_riscv::Reg;
    using FReg = Xbyak_riscv::FReg;

    // Tile registers: 24-register acc/input/index space at v8..v31. With LMUL>1
    // (f32 nspc forward inference only) each logical tile vector is a group of
    // lmul_ consecutive physical registers, so tile indices are spaced by lmul_.
    static constexpr int tile_base = 8;
    static constexpr int tile_size = 24;
    Vmm vreg(int idx) const { return Vmm(tile_base + idx * lmul_); }
    static int reg_ind(int shift, int j, int ur_w) { return shift * ur_w + j; }
    // Effective channel-vector LMUL (1, 2, or 4). 1 everywhere except the f32 nspc
    // forward-inference path, where the channel vector is widened (c_block grows
    // past vlen/32) to amortise the per-channel-block call count. Derived in the
    // ctor from jpp.c_block.
    int lmul_ = 1;

    // Reserved scratch (see class comment).
    const Vmm v_mask = Vmm(0);
    const Vmm v_eltw0 = Vmm(1), v_eltw1 = Vmm(2), v_eltw2 = Vmm(3);
    const Vmm v_bin_rhs = Vmm(4);
    const Vmm v_tmp = Vmm(5); // init value / avg divisor / area / conversions
    const Vmm v_one = Vmm(6); // integer 1 (index increment, training)
    const Vmm v_k_offset = Vmm(7); // running kernel-index (training/backward)

    // GPRs (callee-saved s* are preserved by the manual preamble).
    const Reg reg_param = Xbyak_riscv::a0;
    const Reg reg_input = Xbyak_riscv::s0;
    const Reg reg_output = Xbyak_riscv::s1;
    const Reg reg_index = Xbyak_riscv::s2;
    const Reg reg_kh = Xbyak_riscv::s3; // kh_padding count
    const Reg reg_k_shift = Xbyak_riscv::s4; // kh_padding_shift (index base)
    const Reg reg_kd_pad_shift = Xbyak_riscv::s5; // 3d backward index shift
    const Reg reg_bc = Xbyak_riscv::s6; // b_c (channel-block index)
    const Reg aux_reg_input = Xbyak_riscv::s7;
    const Reg aux_reg_input_d = Xbyak_riscv::s8;
    const Reg reg_kd = Xbyak_riscv::s9; // 3d kd loop counter
    const Reg reg_rhs = Xbyak_riscv::s10; // binary post-op rhs ptr array
    const Reg reg_vl = Xbyak_riscv::s11; // channel AVL for this call

    // FP scratch.
    const FReg f_tmp = Xbyak_riscv::ft0;
    const FReg f_area
            = Xbyak_riscv::ft1; // ker_area_h (avg_exclude divisor base)
    const FReg f_eltw0 = Xbyak_riscv::fa4; // eltwise injector scratch
    const FReg f_eltw1 = Xbyak_riscv::fa5;
    const FReg f_bin = Xbyak_riscv::fa6; // binary injector scalar scratch

    int prev_kw = 0;

    // Set the active vl for this call's channel block (c_block or, for the nspc
    // tail block, c_tail) at SEW=e32/m1. Uses reg_vl (precomputed in generate()).
    void set_vl_e32();

    // Load/store one c_block-wide vector at reg_ptr+offset (bytes). f16 widens on
    // load / narrows on store; the accumulator group is always f32/m1.
    void load(int idx, const Reg &reg_ptr, int offset);
    void store(int idx, const Reg &reg_ptr, int offset);
    void load_indices(int idx, const Reg &reg_ptr, int offset);
    void store_indices(int idx, const Reg &reg_ptr, int offset);

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r);
    void avg_step(int ur_w, int pad_l, int pad_r);
    void max_step_fwd(int ur_w, int pad_l, int pad_r);
    void max_step_bwd(int ur_w, int pad_l, int pad_r);
    void zero_diff_src();

    void step(int ur_w, int pad_l, int pad_r) {
        if (jpp.alg == alg_kind::pooling_max) {
            if (jpp.is_backward)
                max_step_bwd(ur_w, pad_l, pad_r);
            else
                max_step_fwd(ur_w, pad_l, pad_r);
        } else
            avg_step(ur_w, pad_l, pad_r);
    }

    void apply_postops(int ur_w, int c_off);

    void generate() override;

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    // Load a float constant into an FReg via a GPR (reg materialization).
    void load_f32_const(const FReg &f, float v, const Reg &gpr);
    // dst = base + off (bytes), using addi for small offsets, li+add otherwise.
    void addr_off(const Reg &dst, const Reg &base, int off);

    // Far conditional branches. RISC-V B-type branches reach only +/-4KiB; a
    // large loop body (fused post-op chain, big window, wide zeroing) can exceed
    // that and make generate() throw. These emit the branch as a short skip over
    // a J-type jump (+/-1MiB reach).
    void beqz_far(const Reg &r, Xbyak_riscv::Label &t);
    void bnez_far(const Reg &r, Xbyak_riscv::Label &t);
    void blt_far(const Reg &a, const Reg &b, Xbyak_riscv::Label &t);

    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;
};

// Retained native rv64 forward kernel for the nspc and ncsp (plain) layouts,
// used for both inference and training (blocked uses the baked kernel). This
// direct vectorization is preferred over the x64/aarch64 baked port on the plain
// layouts. ncsp vectorizes along OW (interior) or along C (boundary / OW==1);
// nspc vectorizes along C (f16 via generate_f16's n_pos batching, f32 via the
// interior kernel). Shape-agnostic (window bounds via jit_uni_pool_ncsp_args_t),
// VLA (vsetvli).
template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_pool_ncsp_kernel_t : public jit_generator_t {

    jit_uni_pool_ncsp_kernel_t(const jit_pool_conf_t &jpp);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_ncsp_kernel_t)

    // Populate jpp for the native path (nspc or ncsp). Returns unimplemented for
    // any other layout (blocked -> baked kernel).
    static status_t init_conf(jit_pool_conf_t &jpp, primitive_attr_t &attr,
            const pooling_pd_t *ppd);

    void operator()(const jit_uni_pool_ncsp_args_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    jit_pool_conf_t jpp_;
    bool is_max_pool_;
    void generate_f32();
    // f16 path: max accumulates in f16; avg widens to f32 (vfwadd_wv), scales,
    // and narrows back to f16 (vfncvt). Requires the zvfh extension. Eltwise and
    // (at f32) binary post-ops are fused here.
    void generate_f16();
};

// Native gather backward kernel for the nspc and ncsp (plain) layouts (f32 via
// the v ISA, f16 via zvfh). Backward pooling is a gather: the driver enumerates,
// per input position, the output positions whose window covers it (see
// jit_uni_pool_bwd_contrib_t) and this kernel accumulates their diff_dst channel
// rows into the input's diff_src row, then stores it once. max adds diff_dst only
// where the stored argmax matches; avg adds diff_dst * (1/num_summands). f16
// accumulates in f32. Channels are the vector dim (VLA vsetvli); unit-stride for
// nspc, strided (vlse/vsse) for ncsp. No post-ops (backward has default attrs).
template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_pool_bwd_kernel_t : public jit_generator_t {

    jit_uni_pool_bwd_kernel_t(const jit_pool_conf_t &jpp);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_bwd_kernel_t)

    // Populate jpp for the native backward path (nspc or ncsp). Returns
    // unimplemented for blocked (-> baked kernel) and for windows whose
    // per-input covering-output count could exceed the contribution cap.
    static status_t init_conf(jit_pool_conf_t &jpp, const pooling_pd_t *ppd);

    void operator()(const jit_uni_pool_bwd_args_t *p) const {
        jit_generator_t::operator()(p);
    }

    // Upper bound on covering outputs per input = ceil(KD/SD)*ceil(KH/SH)*
    // ceil(KW/SW). The driver builds a stack array of this many contributions;
    // init_conf declines windows that would exceed the cap.
    static constexpr int max_contrib = 512;

protected:
    void generate() override;

private:
    jit_pool_conf_t jpp_;
    bool is_max_pool_;
    void generate_f32();
    void generate_f16();
};

// Shape-baked interior kernel for the full-window interior of an nspc row (f32
// only; f16 uses generate_f16's n_pos batching). kw/stride_w/ur_w, the algorithm,
// the eltwise chain, and the avg_include scale are baked; ur_w output columns
// share each loaded input vector (ARM max_step/avg_step style). Base pointers,
// the VLA channel count, runtime H/D extents, element strides, and the
// avg_exclude scale come in via jit_uni_pool_interior_args_t.
template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_pool_interior_kernel_t : public jit_generator_t {

    jit_uni_pool_interior_kernel_t(const jit_pool_conf_t &ajpp);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_interior_kernel_t)

    void operator()(const jit_uni_pool_interior_args_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    jit_pool_conf_t jpp_;
    void generate_nspc();
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_POOL_KERNEL_HPP
