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

#include <assert.h>
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/jit_rvv_1x1_conv_kernel.hpp"

#define GET_OFF(field) \
    static_cast<int32_t>(offsetof(jit_1x1_conv_args_t, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;
using namespace Xbyak_riscv;

jit_rvv_1x1_conv_kernel_t::jit_rvv_1x1_conv_kernel_t(
        const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jit_generator_t("jit_rvv_1x1_conv_kernel"), jcp(ajcp), attr_(attr) {
    create_kernel();
}

status_t jit_rvv_1x1_conv_kernel_t::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    const int ndims = src_d.ndims();

    jcp.prop_kind = cd.prop_kind;
    jcp.nthr = nthreads;

    // src/weights dtype. f32 multiplies in place; bf16/f16 use widening FMA
    // (Zvfbfwma/Zvfh) into f32 accumulators, so the OC lane count (simd_w,
    // computed from f32 below) is unchanged and only the input element size
    // differs.
    jcp.src_dt = src_d.data_type();

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    // Initialize dimensions
    jcp.mb = src_d.dims()[0];
    jcp.ngroups
            = weights_d.ndims() == src_d.ndims() + 1 ? weights_d.dims()[0] : 1;
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic = jcp.ic_without_padding;

    // Targeting SEW=32 (float), LMUL=1, VLEN <= 1024
    const int vlen = nstl::min(1024u, get_platform_vlen());
    jcp.simd_w = vlen / (sizeof(float) * 8);

    // bf16/f16 accumulate at e32/m2 (2 registers), so one widening FMA spans
    // 2x simd_w OC lanes; f32 uses e32/m1 (simd_w lanes).
    const bool is_lowp
            = utils::one_of(jcp.src_dt, data_type::bf16, data_type::f16);
    const int oc_block = is_lowp ? 2 * jcp.simd_w : jcp.simd_w;

    // OC is padded to match oc_block in weights format (OihwXo)
    // IC is not padded; kernel handles IC tail processing
    jcp.oc = rnd_up(jcp.oc, oc_block);

    // 3D convolution support
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    // Spatial dimensions: D*H*W
    jcp.os = jcp.od * jcp.oh * jcp.ow;
    jcp.is = jcp.id * jcp.ih * jcp.iw;

    jcp.oc_block = oc_block;
    jcp.ic_block = jcp.simd_w;

    // Dynamic parameter calculation
    // Register constraint: (ur * load_loop_blk) + (unroll * load_loop_blk) + 1 <= 32
    jcp.reduce_loop_unroll = 4;

    const int SMALL_SPATIAL = 10;
    const int BIG_SPATIAL = 65;
    const int BIG_LOAD_DIM = (jcp.ic >= 512) ? 256 : 512;

    // Initial load_loop_blk selection
    if (jcp.oc % (2 * jcp.oc_block) == 0 && jcp.os >= 11) {
        jcp.load_loop_blk = 2;
    } else {
        jcp.load_loop_blk = 1;
    }

    // Dynamic ur selection algorithm
    int max_regs, min_regs, size_threshold;

    const int spatial = jcp.od * jcp.oh;

    // Select register range based on batch size and thread count
    if ((8 * jcp.mb) / jcp.nthr >= 1 || jcp.mb == 1) {
        max_regs = 9;
        min_regs = 6;
        size_threshold = 14;

        // Special shape optimization
        if (jcp.oc > 128 && jcp.oc < BIG_LOAD_DIM && spatial > SMALL_SPATIAL
                && spatial < BIG_SPATIAL && jcp.ic < 256) {
            max_regs = 6;
            min_regs = 5;
        }
    } else {
        max_regs = 30;
        min_regs = 9;
        size_threshold = 14;
    }

    // Initial ur
    jcp.ur = 1;

    // First pass: find largest ur that divides spatial evenly
    for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
        if ((spatial >= size_threshold && spatial % ur_w == 0)
                || (spatial < size_threshold && jcp.os % ur_w == 0)) {
            jcp.ur = ur_w;
            break;
        }
    }

    // If first pass fails, use heuristic
    if (jcp.ur == 1) {
        jcp.ur = nstl::min(max_regs, jcp.os);
        int os_tail = jcp.os % max_regs;
        for (int i = max_regs; i >= min_regs; i--) {
            int i_tail = jcp.os % i;
            if (i_tail > os_tail || i_tail == 0) {
                jcp.ur = i;
                os_tail = i_tail;
                if (i_tail == 0) break;
            }
        }
    }

    // Adjust ur based on load_loop_blk (ensure register limit).
    // f32:      1 (v0) + ur*blk accums (m1) + unroll*blk weights (m1) <= 32.
    // bf16/f16: 2 (v0,v1 pad) + 2*ur*blk accums (m2) + 2 bias temp (m2)
    //           + unroll*blk weights (m1) <= 32.
    const int regs_per_accum = is_lowp ? 2 : 1;
    const int reg_avail = is_lowp ? (32 - 2 - 2) : (32 - 1);
    int max_ur_for_blk
            = (reg_avail - jcp.reduce_loop_unroll * jcp.load_loop_blk)
            / (regs_per_accum * jcp.load_loop_blk);
    if (jcp.ur > max_ur_for_blk) {
        jcp.ur = max_ur_for_blk;
        if (jcp.ur < 1) jcp.ur = 1;
    }

    jcp.load_block = jcp.oc_block;
    jcp.reduce_block = jcp.ic_block;

    jcp.bcast_block = jcp.ur;
    jcp.load_dim = jcp.oc_without_padding;
    jcp.bcast_dim = jcp.os;
    jcp.reduce_dim = jcp.ic_without_padding;

    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast = div_up(jcp.os, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.oc_without_padding, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.ic_without_padding, jcp.reduce_block);
    jcp.load_grp_count = 1;

    // Blocking strategy for NHWC layout
    jcp.nb_reduce_blocking = jcp.nb_reduce;
    jcp.nb_load_blocking = jcp.nb_load;
    jcp.nb_load_blocking_max = jcp.nb_load;

    // Spatial dimension blocking (in ur units)
    int target_bcast_blocking = 735;
    jcp.nb_bcast_blocking
            = nstl::min(jcp.nb_bcast, div_up(target_bcast_blocking, jcp.ur));
    if (jcp.nb_bcast_blocking == 0) jcp.nb_bcast_blocking = 1;
    jcp.nb_bcast_blocking_max = jcp.nb_bcast_blocking;

    // Optimize reduce_loop_unroll based on available registers
    if (jcp.load_loop_blk == 2) {
        jcp.reduce_loop_unroll = 4;
    } else {
        jcp.reduce_loop_unroll = 4;
    }

    // Layout-dependent stride parameters (for NHWC). Inputs (src/weights) may
    // be 2-byte (bf16/f16); the destination accumulates/stores in f32.
    jcp.typesize_in = static_cast<int>(types::data_type_size(jcp.src_dt));
    jcp.typesize_out = sizeof(float);

    jcp.reduce_loop_bcast_step = jcp.typesize_in;
    jcp.reduce_loop_load_step = jcp.oc_block * jcp.typesize_in;

    // Strides within bcast_loop (spatial dimensions)
    jcp.bcast_loop_bcast_step
            = jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    jcp.bcast_loop_output_step
            = jcp.ngroups * jcp.oc_without_padding * jcp.typesize_out;

    // Strides within load_loop (OC dimension)
    jcp.load_loop_load_step
            = jcp.ic_without_padding * jcp.oc_block * jcp.typesize_in;
    jcp.load_loop_iter_step = jcp.oc_block;

    return status::success;
}

void jit_rvv_1x1_conv_kernel_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {
    // Not implemented
}

void jit_rvv_1x1_conv_kernel_t::balance(jit_1x1_conv_conf_t &jcp) {
    // Not implemented
}

void jit_rvv_1x1_conv_kernel_t::generate() {
    preamble();

    // Accumulator view: e32/m2 for bf16/f16 (2 regs per accum), e32/m1 for f32.
    const auto acc_lmul
            = is_lowp() ? Xbyak_riscv::LMUL::m2 : Xbyak_riscv::LMUL::m1;

    // Set initial VL to oc_block
    li(reg_tmp_imm, jcp.oc_block);
    vsetvli(reg_tmp_imm, reg_tmp_imm, SEW::e32, acc_lmul, VTA::ta, VMA::ma);

    // Load parameters
    ld(reg_bcast_data, reg_param, GET_OFF(bcast_data));
    ld(reg_load_data, reg_param, GET_OFF(load_data));
    ld(reg_output_data, reg_param, GET_OFF(output_data));
    if (jcp.with_bias) ld(reg_bias_data, reg_param, GET_OFF(bias_data));

    ld(reg_load_loop_work, reg_param, GET_OFF(load_dim));
    ld(reg_bcast_loop_work, reg_param, GET_OFF(bcast_dim));
    ld(reg_reduce_loop_work, reg_param, GET_OFF(reduce_dim));
    ld(reg_reduce_pos_flag, reg_param, GET_OFF(first_last_flag));

    // Main loop generation
    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);

        // Update pointers and work counters
        li(reg_tmp_imm, load_loop_blk * jcp.load_loop_load_step);
        add(reg_load_data, reg_load_data, reg_tmp_imm);

        if (jcp.with_bias) {
            li(reg_tmp_imm, load_loop_blk * jcp.oc_block * jcp.typesize_out);
            add(reg_bias_data, reg_bias_data, reg_tmp_imm);
        }

        li(reg_tmp_imm, load_loop_blk * jcp.oc_block * jcp.typesize_out);
        add(reg_output_data, reg_output_data, reg_tmp_imm);

        li(reg_tmp_imm, load_loop_blk * jcp.load_loop_iter_step);
        sub(reg_load_loop_work, reg_load_loop_work, reg_tmp_imm);
    };

    Label load_loop_label, load_loop_end, load_loop_tail;

    if (jcp.load_loop_blk > 1) {
        L(load_loop_label);
        li(reg_tmp_imm, jcp.load_loop_blk * jcp.oc_block);
        blt(reg_load_loop_work, reg_tmp_imm, load_loop_tail);

        // Ensure VL is full
        li(reg_tmp_imm, jcp.oc_block);
        vsetvli(reg_tmp_imm, reg_tmp_imm, SEW::e32, acc_lmul, VTA::ta, VMA::ma);
        mv(reg_blk_vl, reg_tmp_imm); // save VL for bf16/f16 SEW switches

        load_loop_body(jcp.load_loop_blk);
        jal(x0, load_loop_label);
    }

    L(load_loop_tail);
    {
        Label tail_loop;
        L(tail_loop);
        blez(reg_load_loop_work, load_loop_end);

        // Last block may be partial, use vsetvli to set VL dynamically
        vsetvli(reg_tmp_imm, reg_load_loop_work, SEW::e32, acc_lmul, VTA::ta,
                VMA::ma);
        mv(reg_blk_vl, reg_tmp_imm); // save VL for bf16/f16 SEW switches

        bcast_loop(1);

        // Update pointers and work counters (tail loop)
        li(reg_tmp_imm, jcp.load_loop_load_step);
        add(reg_load_data, reg_load_data, reg_tmp_imm);
        if (jcp.with_bias) {
            li(reg_tmp_imm, jcp.oc_block * jcp.typesize_out);
            add(reg_bias_data, reg_bias_data, reg_tmp_imm);
        }
        li(reg_tmp_imm, jcp.oc_block * jcp.typesize_out);
        add(reg_output_data, reg_output_data, reg_tmp_imm);

        li(reg_tmp_imm, jcp.oc_block);
        sub(reg_load_loop_work, reg_load_loop_work, reg_tmp_imm);

        jal(x0, tail_loop);
    }
    L(load_loop_end);

    postamble();
}

void jit_rvv_1x1_conv_kernel_t::preamble() {
    addi(sp, sp, -64);
    sd(ra, sp, 56);
    sd(s0, sp, 48);
    sd(s1, sp, 40);
    sd(s2, sp, 32);
    sd(s3, sp, 24);
    sd(s4, sp, 16);
    sd(s5, sp, 8);
}

void jit_rvv_1x1_conv_kernel_t::postamble() {
    ld(ra, sp, 56);
    ld(s0, sp, 48);
    ld(s1, sp, 40);
    ld(s2, sp, 32);
    ld(s3, sp, 24);
    ld(s4, sp, 16);
    ld(s5, sp, 8);
    addi(sp, sp, 64);
    ret();
}

void jit_rvv_1x1_conv_kernel_t::bcast_loop(int load_loop_blk) {
    mv(reg_bcast_loop_iter, reg_bcast_loop_work);
    mv(aux1_reg_bcast_data, reg_bcast_data);
    mv(aux_reg_output_data, reg_output_data);

    Label bcast_loop_label, bcast_loop_tail;

    li(reg_tmp_imm, jcp.ur);
    blt(reg_bcast_loop_iter, reg_tmp_imm, bcast_loop_tail);

    L(bcast_loop_label);
    {
        reduce_loop(load_loop_blk, jcp.ur);

        li(reg_tmp_imm, jcp.ur * jcp.bcast_loop_bcast_step);
        add(aux1_reg_bcast_data, aux1_reg_bcast_data, reg_tmp_imm);

        li(reg_tmp_imm, jcp.ur * jcp.bcast_loop_output_step);
        add(aux_reg_output_data, aux_reg_output_data, reg_tmp_imm);

        addi(reg_bcast_loop_iter, reg_bcast_loop_iter, -jcp.ur);
        li(reg_tmp_imm, jcp.ur);
        bge(reg_bcast_loop_iter, reg_tmp_imm, bcast_loop_label);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail > 0) {
        Label bcast_loop_tail_end;
        blez(reg_bcast_loop_iter, bcast_loop_tail_end);

        reduce_loop(load_loop_blk, jcp.ur_tail);

        L(bcast_loop_tail_end);
    }
}

void jit_rvv_1x1_conv_kernel_t::reduce_loop(int load_loop_blk, int ur) {
    mv(aux_reg_load_data, reg_load_data);
    mv(aux_reg_bcast_data, aux1_reg_bcast_data);

    // bf16/f16: load src (broadcast) and weights as 16-bit and accumulate via
    // widening FMA into the f32 accumulators. The accumulator init/store and
    // bias run at e32/m2; the weight-load + FMA region runs at e16/m1 (same VL,
    // since VLMAX(e16,m1) == VLMAX(e32,m2) == oc_block). f32 stays e32/m1 and
    // is unchanged.
    const bool is_lowp
            = utils::one_of(jcp.src_dt, data_type::bf16, data_type::f16);
    const bool is_bf16 = jcp.src_dt == data_type::bf16;

    auto emit_wload = [=](const VReg &vr, const Reg &addr) {
        if (is_lowp)
            vle16_v(vr, addr);
        else
            vle32_v(vr, addr);
    };
    auto emit_bcast = [=](const FReg &fr, const Reg &addr, int off) {
        if (is_lowp)
            flh(fr, addr, off);
        else
            flw(fr, addr, off);
    };
    auto emit_fma = [=](const VReg &acc, const FReg &fr, const VReg &vr) {
        if (is_bf16)
            vfwmaccbf16_vf(acc, fr, vr);
        else if (is_lowp)
            vfwmacc_vf(acc, fr, vr);
        else
            vfmacc_vf(acc, fr, vr);
    };
    auto sew_compute = [=]() {
        if (is_lowp)
            vsetvli(reg_tmp_imm, reg_blk_vl, SEW::e16, LMUL::m1, VTA::ta,
                    VMA::ma);
    };
    auto sew_accum = [=]() {
        if (is_lowp)
            vsetvli(reg_tmp_imm, reg_blk_vl, SEW::e32, LMUL::m2, VTA::ta,
                    VMA::ma);
    };

    auto init = [=]() {
        Label init_zero, init_done;
        andi(reg_tmp_imm, reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
        bnez(reg_tmp_imm, init_zero);

        // Load from dst for accumulation
        mv(reg_tmp_addr, aux_reg_output_data);
        for (int i_ur = 0; i_ur < ur; ++i_ur) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                vle32_v(vreg_accum(i_load, i_ur), reg_tmp_addr);
                if (i_load + 1 < load_loop_blk)
                    addi(reg_tmp_addr, reg_tmp_addr,
                            jcp.load_block * jcp.typesize_out);
            }
            li(reg_tmp_imm,
                    jcp.bcast_loop_output_step
                            - (load_loop_blk - 1) * jcp.load_block
                                    * jcp.typesize_out);
            add(reg_tmp_addr, reg_tmp_addr, reg_tmp_imm);
        }
        jal(x0, init_done);

        L(init_zero);
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            if (jcp.with_bias) {
                size_t bias_off
                        = (size_t)i_load * jcp.oc_block * jcp.typesize_out;
                if (bias_off == 0) {
                    vle32_v(vreg_bias_tmp(), reg_bias_data);
                } else {
                    li(reg_tmp_addr, bias_off);
                    add(reg_tmp_addr, reg_tmp_addr, reg_bias_data);
                    vle32_v(vreg_bias_tmp(), reg_tmp_addr);
                }
            }
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (jcp.with_bias) {
                    vmv_v_v(vreg_accum(i_load, i_ur), vreg_bias_tmp());
                } else {
                    vxor_vv(vreg_accum(i_load, i_ur), vreg_accum(i_load, i_ur),
                            vreg_accum(i_load, i_ur));
                }
            }
        }
        L(init_done);
    };

    auto store = [=]() {
        mv(reg_tmp_addr, aux_reg_output_data);
        for (int i_ur = 0; i_ur < ur; ++i_ur) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                vse32_v(vreg_accum(i_load, i_ur), reg_tmp_addr);
                if (i_load + 1 < load_loop_blk)
                    addi(reg_tmp_addr, reg_tmp_addr,
                            jcp.load_block * jcp.typesize_out);
            }
            li(reg_tmp_imm,
                    jcp.bcast_loop_output_step
                            - (load_loop_blk - 1) * jcp.load_block
                                    * jcp.typesize_out);
            add(reg_tmp_addr, reg_tmp_addr, reg_tmp_imm);
        }
    };

    auto fma_block = [=](int current_unroll, bool last_block) {
        for (int i_unroll = 0; i_unroll < current_unroll; ++i_unroll) {
            emit_bcast(freg_bcast, aux_reg_bcast_data, 0);

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    emit_fma(vreg_accum(i_load, i_ur), freg_bcast,
                            vreg_load(i_load, i_unroll));
                }

                if (i_ur + 1 < ur) {
                    size_t offset
                            = (size_t)(i_ur + 1) * jcp.bcast_loop_bcast_step;
                    if (offset <= 2047) {
                        emit_bcast(freg_bcast, aux_reg_bcast_data, (int)offset);
                    } else {
                        li(reg_tmp_addr, offset);
                        add(reg_tmp_addr, reg_tmp_addr, aux_reg_bcast_data);
                        emit_bcast(freg_bcast, reg_tmp_addr, 0);
                    }
                }
            }
            addi(aux_reg_bcast_data, aux_reg_bcast_data,
                    jcp.reduce_loop_bcast_step);
        }

        // Update weight pointer to next unroll block
        li(reg_tmp_imm, jcp.reduce_loop_unroll * jcp.reduce_loop_load_step);
        add(aux_reg_load_data, aux_reg_load_data, reg_tmp_imm);

        // Prefetch weights for next iteration
        if (!last_block) {
            for (int i_unroll = 0; i_unroll < jcp.reduce_loop_unroll;
                    ++i_unroll) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    size_t weight_off
                            = (size_t)i_unroll * jcp.reduce_loop_load_step
                            + (size_t)i_load * jcp.load_loop_load_step;
                    li(reg_tmp_addr, weight_off);
                    add(reg_tmp_addr, aux_reg_load_data, reg_tmp_addr);
                    emit_wload(vreg_load(i_load, i_unroll), reg_tmp_addr);
                }
            }
        }
    };

    init();

    // Accumulators are now seeded (e32); switch to the e16/mf2 source view for
    // the bf16/f16 widening-FMA reduction (no-op for f32).
    sew_compute();

    // Load first round of weights (IC=0..unroll-1)
    for (int i_unroll = 0; i_unroll < jcp.reduce_loop_unroll; ++i_unroll) {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            size_t weight_off = (size_t)i_unroll * jcp.reduce_loop_load_step
                    + (size_t)i_load * jcp.load_loop_load_step;
            if (weight_off == 0) {
                emit_wload(vreg_load(i_load, i_unroll), aux_reg_load_data);
            } else {
                li(reg_tmp_addr, weight_off);
                add(reg_tmp_addr, aux_reg_load_data, reg_tmp_addr);
                emit_wload(vreg_load(i_load, i_unroll), reg_tmp_addr);
            }
        }
    }

    mv(reduce_loop_iter, reg_reduce_loop_work);
    Label reduce_loop_label, reduce_loop_tail;

    li(reg_tmp_imm, jcp.reduce_loop_unroll);
    blt(reduce_loop_iter, reg_tmp_imm, reduce_loop_tail);

    L(reduce_loop_label);
    {
        li(reg_tmp_imm, jcp.reduce_loop_unroll);
        sub(reg_tmp_imm, reduce_loop_iter, reg_tmp_imm);
        li(reg_tmp_addr, jcp.reduce_loop_unroll);
        Label is_last, do_fma;
        blt(reg_tmp_imm, reg_tmp_addr, is_last);
        fma_block(jcp.reduce_loop_unroll, false);
        jal(x0, do_fma);
        L(is_last);
        fma_block(jcp.reduce_loop_unroll, true);
        L(do_fma);

        addi(reduce_loop_iter, reduce_loop_iter, -jcp.reduce_loop_unroll);
        li(reg_tmp_imm, jcp.reduce_loop_unroll);
        bge(reduce_loop_iter, reg_tmp_imm, reduce_loop_label);
    }

    L(reduce_loop_tail);
    {
        Label tail_done;
        blez(reduce_loop_iter, tail_done);
        Label tail_loop;
        L(tail_loop);
        {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                size_t weight_off = (size_t)i_load * jcp.load_loop_load_step;
                if (weight_off == 0) {
                    emit_wload(vreg_load(i_load, 0), aux_reg_load_data);
                } else {
                    li(reg_tmp_addr, weight_off);
                    add(reg_tmp_addr, aux_reg_load_data, reg_tmp_addr);
                    emit_wload(vreg_load(i_load, 0), reg_tmp_addr);
                }
            }

            emit_bcast(freg_bcast, aux_reg_bcast_data, 0);
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    emit_fma(vreg_accum(i_load, i_ur), freg_bcast,
                            vreg_load(i_load, 0));
                }
                if (i_ur + 1 < ur) {
                    size_t offset
                            = (size_t)(i_ur + 1) * jcp.bcast_loop_bcast_step;
                    if (offset <= 2047) {
                        emit_bcast(freg_bcast, aux_reg_bcast_data, (int)offset);
                    } else {
                        li(reg_tmp_addr, offset);
                        add(reg_tmp_addr, reg_tmp_addr, aux_reg_bcast_data);
                        emit_bcast(freg_bcast, reg_tmp_addr, 0);
                    }
                }
            }

            addi(aux_reg_bcast_data, aux_reg_bcast_data,
                    jcp.reduce_loop_bcast_step);
            addi(aux_reg_load_data, aux_reg_load_data,
                    jcp.reduce_loop_load_step);
            addi(reduce_loop_iter, reduce_loop_iter, -1);
            bnez(reduce_loop_iter, tail_loop);
        }
        L(tail_done);
    }

    // Restore the e32/m1 accumulator view before storing the f32 results.
    sew_accum();

    store();
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
