/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_RV64_JIT_UNI_BINARY_KERNEL_HPP
#define CPU_RV64_JIT_UNI_BINARY_KERNEL_HPP

#include <cassert>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_primitive_conf.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

struct binary_kernel_t : public jit_generator_t {
    using op_t = binary_op_t;
    using bcast_t = binary_bcast_t;

    binary_kernel_t(const binary_pd_t *pd, const jit_binary_conf_t &conf,
            const char *name, bool tail_kernel = false)
        : jit_generator_t(name)
        , pd_(pd)
        , conf_(conf)
        , is_tail_kernel_(tail_kernel)
        , tail_size_(get_tail_size()) {}
    ~binary_kernel_t() override = default;

    void operator()(const jit_uni_binary_args_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    // The c_blocked channel tail (x64's get_tail_size specialized to the
    // driver contract: the tail kernel exists for c_blocked oc tails only).
    dim_t get_tail_size() const {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        return conf_.op_type == op_t::c_blocked && !src0_d.is_plain()
                ? src0_d.dims()[1] % src0_d.blocking_desc().inner_blks[0]
                : 0;
    }

    // Unlike x64/aarch64 there is no vlen_/simd_w_/tail-mask machinery: the
    // generated code is vector-length-agnostic and a single vsetvli loop
    // covers the plain-shape element tail. The tail KERNEL (x64 shape) exists
    // only for the blocked channel tail: it stores the valid lanes of each
    // channel block and zeroes the padded tail.
    const binary_pd_t *pd_;
    const jit_binary_conf_t conf_;
    const bool is_tail_kernel_;
    const dim_t tail_size_;
};

// Vector-length-agnostic binary kernel. A single class rather than the
// x64/aarch64 <isa> template family: RVV has one scalable vector ISA and the
// generated code adapts to the hardware VLEN at run time.
//
// All arithmetic runs in the f32 e32/m4 compute domain; f16/bf16 operands are
// widened/narrowed at the load/store edge (LMUL pair e16/m2).
// The dtype conversions x64 delegates to io_multi_dt_helper_t are emitted
// directly by load_vector/store_vector/load_scalar. A broadcast (scalar) src1
// stays in a scalar FP register and feeds the .vf instruction forms instead of
// being broadcast into a vector register as on x64; a strided src1 (different
// plain src0/src1 layouts, e.g. nchw:nhwc) is read with vlse instead of x64's
// index-vector gather. Ternary select folds into the same path via the v0 mask.
struct jit_uni_binary_kernel_t : public binary_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binary_kernel_t)

    // --- compute vtype ---
    // The f32 compute domain is e32 at compute_lmul_. m4 is NOT a taste
    // choice, it is load-bearing on the two paths below; changing it requires
    // re-deriving both invariants.
    //   1. The c_blocked block-stepped loop uses a constant AVL of c_blk and
    //      advances the pointers by that same constant, so the vsetvli must
    //      grant it in full: VLMAX(e32, compute_lmul_) >= c_blk (<= 16).
    //   2. zero_pad_tail() writes the padded lanes of one block in a single
    //      store: VLMAX(e8, compute_lmul_) >= (16 - 1) * sizeof(f32).
    // Both hold at the smallest legal vector length, and the V extension
    // implies Zvl128b, so VLEN >= min_vlen_ is architecturally guaranteed.
    static constexpr int min_vlen_ = 128;
    static constexpr size_t compute_group_stride_ = 4; // registers per acc
    static constexpr LMUL compute_lmul_ = LMUL::m4;
    // Narrow staging LMULs, chosen to share the compute VLMAX so that
    // `vsetvli x0, x0` preserves vl across the widen/narrow (see load_vector).
    static constexpr LMUL stage_e16_lmul_ = LMUL::m2;
    static constexpr LMUL stage_e8_lmul_ = LMUL::m1;
    static_assert(compute_group_stride_ * (min_vlen_ / 32) >= 16,
            "VLMAX(e32, compute_lmul_) must cover a full 16-channel block");
    static_assert(
            compute_group_stride_ * (min_vlen_ / 8) >= 15 * (int)sizeof(float),
            "VLMAX(e8, compute_lmul_) must cover one block's zero padding");

    // Register map (e32/m4 compute): a1..a5 hold the unpacked call arguments
    // (a0 stays live as the args pointer -- the binary post-op injector reads
    // the rhs pointer array and dst_orig through it, as x64 reads through
    // abi_param1), t0 = vl granted by vsetvli, t1/t2/t3 = byte-stride/address
    // scratch (t1/t3 double as the binary post-op injector's address scratch;
    // t4/t5/t6/a6/a7 are the injector's fixed X_TMP scratch, preserved around
    // each injection via preserve_gpr_helpers as on x64).
    // v4 = src0/accumulator, v8 = src1, v12/v16 = xf16 staging (also the sum
    // old-dst and select condition staging), v20 = binary post-op rhs;
    // v8/v12/v16 + v24/v28 are the five eltwise-injector aux groups (all dead
    // around the injector call), v0 is the mask register.
    const Reg reg_param_ = a0; // no abi_param aliases on rv64
    const Reg reg_src0_ = a1;
    const Reg reg_src1_ = a2;
    const Reg reg_dst_ = a3;
    const Reg reg_work_amount_ = a4;
    const Reg reg_src2_ = a5;
    const Reg reg_vl_ = t0;
    const Reg reg_bytes_ = t1;
    const Reg reg_tmp_ = t2;
    const Reg reg_tmp1_ = t3;
    // Block-stepped AVL constant (c_block, or the valid tail lanes in the
    // tail kernel), and the per-run remaining count in the different-layouts
    // loop -- the two modes never coexist. Doubles as injector scratch, kept
    // alive across injections by the preserve guard.
    const Reg reg_blk_avl_ = t4;
    // Hoisted src1 byte stride for the different-layouts vlse (x64 keeps it
    // in the indices vector instead); guarded like reg_blk_avl_.
    const Reg reg_src1_stride_ = t5;
    // v0 is the architecturally fixed mask register (vmerge/vfmerge only read
    // v0.t), unlike x64's assignable k-masks.
    const VReg vreg_mask_ = VReg(0);
    const VReg vreg_src0_ = VReg(4);
    const VReg vreg_src1_ = VReg(8);
    const VReg vreg_tmp0_ = VReg(12);
    const VReg vreg_tmp1_ = VReg(16);
    const VReg vreg_rhs_ = VReg(20);
    const FReg freg_tmp_ = fa2; // integer saturation bounds (dead post-inject)
    const FReg freg_bcast_src1_ = fa3;
    const FReg freg_scales_src0_ = fa4;
    const FReg freg_scales_src1_ = fa5;
    const FReg freg_sum_scale_ = fa6;

    // The postops injector's isa selects its is_data_supported contract -- in
    // particular whether an xf16 binary rhs is a legal post-op operand.
    // It must therefore match the isa the pd validated the chain with
    // (post_ops_ok(mayiuse(zvfh) ? zvfh : v)); otherwise a zvfh machine accepts
    // an f16 rhs that the <v> injector then rejects (an is_data_supported assert
    // fires at codegen time). RVV codegen is vector-length-agnostic and
    // identical across v/zvfh, so the single non-templated kernel just holds
    // whichever instantiation matches the runtime isa (x64 templates the whole
    // kernel on isa instead). Exactly one is set; see init_post_ops_injector.
    std::unique_ptr<injector::jit_uni_postops_injector_t<v>>
            postops_injector_v_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<zvfh>>
            postops_injector_zvfh_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<zvfbfwma>>
            postops_injector_zvfbfwma_;
    bool has_postops_injector() const {
        return postops_injector_v_ || postops_injector_zvfh_
                || postops_injector_zvfbfwma_;
    }

    void init();
    void init_post_ops_injector();
    void apply_postops();
    void load_kernel_params();
    // v0 mask -> f32 0.0/1.0 in the accumulator (comparison result contract).
    void materialize_cmp();
    void perform_op(const VReg &v0, const VReg &v1);
    void perform_op(const VReg &v0, const FReg &s1);
    void perform_ternary_op(
            const VReg &v0, const VReg &v1, const VReg &vreg_stage);
    void compute_bcast();
    // Load vl elements of dtype dt from reg_ptr into the f32 group vreg
    // (e32/m4), staging narrow dtypes in vreg_stage. Ends at the compute
    // vtype. stride_elems > 1 reads with an element stride (vlse).
    void load_vector(data_type_t dt, const VReg &vreg, const VReg &vreg_stage,
            const Reg &reg_ptr, dim_t stride_elems = 1);
    // Load one scalar of dtype dt from reg_ptr as f32 into freg.
    void load_scalar(data_type_t dt, const FReg &freg, const Reg &reg_ptr);
    // Convert the f32 result group (e32/m4) to dt and store vl elements.
    void store_vector(data_type_t dt, const VReg &vreg, const VReg &vreg_stage,
            const Reg &reg_ptr);
    void compute_dst_body();
    void compute_dst();
    void forward();
    // The different-layouts loop: runs of outer_dims strided-src1 elements
    // with a per-run src1 rebase (x64's stride-range reset, with outer_dims/
    // src1_stride baked as codegen constants).
    void forward_over_outer_dims();
    void generate() override;
    // --- rv64-specific low-level helpers (no x64 method analog; cf. aarch64,
    // which declares its arch-specific helpers after generate()) ---
    // Set vl for the remaining work at the e32/m4 compute vtype.
    void set_compute_vl();
    // Reissue the e32/m4 compute vtype for the current vl.
    void to_compute_vtype();
    // Materialize an f32 constant into freg via a GPR (RVV .vf/.vx ops take a
    // scalar register directly, so no rodata table -- x64 loads from p_table).
    void load_f32_const(const FReg &freg, float val);
    // Tail kernel: zero the padded lanes [tail, c_block) of the current block
    // (x64's tail-kernel zero-padding store).
    void zero_pad_tail(int c_blk);

    jit_uni_binary_kernel_t(const binary_pd_t *pd,
            const jit_binary_conf_t &conf, bool tail_kernel = false);
    ~jit_uni_binary_kernel_t() override = default;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
