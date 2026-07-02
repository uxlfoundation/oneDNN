/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#ifndef CPU_RV64_JIT_RVV_1X1_CONV_KERNEL_HPP
#define CPU_RV64_JIT_RVV_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

struct jit_rvv_1x1_conv_kernel_t : public jit_generator_t {
    jit_rvv_1x1_conv_kernel_t(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_1x1_conv_kernel)

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
            int nthreads, bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp);

    static void balance(jit_1x1_conv_conf_t &jcp);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    using Reg = Xbyak_riscv::Reg;
    using VReg = Xbyak_riscv::VReg;
    using FReg = Xbyak_riscv::FReg;

    const Reg reg_param = a0;
    const Reg reg_bcast_data = a1;
    const Reg reg_load_data = a2;
    const Reg reg_output_data = a3;
    const Reg reg_bias_data = a4;

    const Reg reg_load_loop_work = t0;
    const Reg reg_bcast_loop_work = t1;
    const Reg reg_reduce_loop_work = t2;

    const Reg aux_reg_bcast_data = t3;
    const Reg aux_reg_load_data = t4;
    const Reg aux_reg_output_data = t5;
    const Reg aux1_reg_bcast_data = t6;

    const Reg reduce_loop_iter = s0;
    const Reg reg_bcast_loop_iter = s1;
    const Reg reg_reduce_pos_flag = s2;
    // Saved active vector length (in OC lanes) for the current load block, so
    // the bf16/f16 path can switch SEW (e32<->e16) without changing VL.
    const Reg reg_blk_vl = s3;

    const Reg reg_tmp_imm = s4;
    const Reg reg_tmp_addr = s5;

    // bf16/f16 use e32/m2 accumulators (2 registers each, even-aligned) fed by
    // e16/m1 weights, so one widening FMA covers 2x the OC lanes of the f32
    // e32/m1 path. f32 keeps single-register (m1) accumulators from v1.
    bool is_lowp() const {
        return jcp.src_dt == data_type::bf16 || jcp.src_dt == data_type::f16;
    }
    int acc_nregs() const { return is_lowp() ? 2 : 1; }
    int acc_base() const { return is_lowp() ? 2 : 1; }

    VReg vreg_accum(int i_load, int i_ur) {
        return VReg(
                acc_base() + acc_nregs() * (i_ur * jcp.load_loop_blk + i_load));
    }

    // Bias staging register: for f32 this aliases vreg_load(0) (unchanged); for
    // bf16/f16 it is a dedicated m2 (2-register) slot right after the accums.
    VReg vreg_bias_tmp() {
        return VReg(acc_base() + acc_nregs() * jcp.ur * jcp.load_loop_blk);
    }

    VReg vreg_load(int i_load, int i_unroll = 0) {
        const int after_acc
                = acc_base() + acc_nregs() * jcp.ur * jcp.load_loop_blk;
        // bf16/f16 reserve a 2-register bias temp between accums and weights.
        const int base = is_lowp() ? after_acc + 2 : after_acc;
        return VReg(base + i_unroll * jcp.load_loop_blk + i_load);
    }

    const FReg freg_bcast = fa0;
    const FReg freg_load = fa1;

    void generate() override;
    void preamble();
    void postamble();
    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur);
    void fma_block(int load_loop_blk, int ur);
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
