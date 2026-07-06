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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/rv64/jit_rvv_1x1_conv_kernel.hpp"
#include "cpu/rv64/rvjit/rvjit.hpp"

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
using namespace rvjit;

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

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.mb = src_d.dims()[0];
    jcp.ngroups
            = weights_d.ndims() == src_d.ndims() + 1 ? weights_d.dims()[0] : 1;
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic = jcp.ic_without_padding;

    // Targeting SEW=32 (float), LMUL=4; oc_block spans one full m4 group
    const int vlen = nstl::min(1024u, get_platform_vlen());
    jcp.simd_w = vlen / (sizeof(float) * 8);
    jcp.oc_block = jcp.simd_w * 4;

    // OC padded to oc_block; IC left unpadded (kernel handles IC tail)
    jcp.oc = rnd_up(jcp.oc, jcp.oc_block);

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.os = jcp.od * jcp.oh * jcp.ow;
    jcp.is = jcp.id * jcp.ih * jcp.iw;

    jcp.ic_block = jcp.simd_w;

    jcp.reduce_loop_unroll = 4;

    const int SMALL_SPATIAL = 10;
    const int BIG_SPATIAL = 65;
    const int BIG_LOAD_DIM = (jcp.ic >= 512) ? 256 : 512;

    // Always process one oc_block at a time; LMUL=4 provides the throughput
    jcp.load_loop_blk = 1;

    int max_regs, min_regs, size_threshold;

    const int spatial = jcp.od * jcp.oh;

    if ((8 * jcp.mb) / jcp.nthr >= 1 || jcp.mb == 1) {
        max_regs = 9;
        min_regs = 6;
        size_threshold = 14;

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

    // Hardware ceiling on independent N-wide accumulator groups
    const int VEC_MAX_UR = rv64_rvjit_model().vpu.max_n_accumulators(
            to_rvjit_sew(jcp.src_dt), to_rvjit_sew(data_type::f32));
    max_regs = nstl::min(max_regs, VEC_MAX_UR);
    min_regs = nstl::min(min_regs, VEC_MAX_UR);

    jcp.ur = 1;

    for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
        if ((spatial >= size_threshold && spatial % ur_w == 0)
                || (spatial < size_threshold && jcp.os % ur_w == 0)) {
            jcp.ur = ur_w;
            break;
        }
    }

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

    jcp.nb_reduce_blocking = jcp.nb_reduce;
    jcp.nb_load_blocking = jcp.nb_load;
    jcp.nb_load_blocking_max = jcp.nb_load;

    int target_bcast_blocking = 735;
    jcp.nb_bcast_blocking
            = nstl::min(jcp.nb_bcast, div_up(target_bcast_blocking, jcp.ur));
    if (jcp.nb_bcast_blocking == 0) jcp.nb_bcast_blocking = 1;
    jcp.nb_bcast_blocking_max = jcp.nb_bcast_blocking;

    jcp.typesize_in = types::data_type_size(jcp.src_dt);
    jcp.typesize_out = sizeof(float); // dst stays f32
    jcp.typesize_acc = sizeof(float); // accumulator stays f32

    jcp.reduce_loop_bcast_step = jcp.typesize_in;
    jcp.reduce_loop_load_step = jcp.oc_block * jcp.typesize_in;

    jcp.bcast_loop_bcast_step
            = jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    jcp.bcast_loop_output_step
            = jcp.ngroups * jcp.oc_without_padding * jcp.typesize_out;

    jcp.load_loop_load_step
            = jcp.ic_without_padding * jcp.oc_block * jcp.typesize_in;
    jcp.load_loop_iter_step = jcp.oc_block;

    return status::success;
}

void jit_rvv_1x1_conv_kernel_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {}

void jit_rvv_1x1_conv_kernel_t::balance(jit_1x1_conv_conf_t &jcp) {}

void jit_rvv_1x1_conv_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1

    const data_type_t dt = jcp.src_dt;
    const data_type_t dt_c = data_type::f32;

    const dim_t lda_bytes = jcp.reduce_loop_load_step;
    const dim_t ldb_bytes = jcp.bcast_loop_bcast_step;
    const dim_t ldc_bytes = jcp.bcast_loop_output_step;
    const dim_t oc_step_w = jcp.load_loop_load_step;

    rvjit_t m(*this);
    m.set_model(rv64_rvjit_model());
    auto &ca = m.const_folding();
    auto &cf = m.control_flow();
    auto &pool = m.register_pool();
    auto &mem = m.memory_move();
    auto &eng = m.matmul_engine();

    // Fixed live registers
    const Reg args = a0;
    const Reg vin_wptr = a1; // A: weights (ptra)
    const Reg ptr_bcast = a2; // B: src    (ptrb)
    const Reg dst_wptr = a3; // C: output (ptrc)

    // Setup register pools
    pool.int_register_file_excluding({args, vin_wptr, ptr_bcast, dst_wptr});

    // Conv-specific OC-loop bookkeeping, not part of the matmul engine
    const const_t vin_oc_step = pool.new_const(oc_step_w);
    const const_t dst_oc_step = pool.new_const(jcp.oc_block * jcp.typesize_out);
    const const_t neg_oc_block_c = pool.new_const(-jcp.oc_block);

    // Setup resources for temporaries and define aliases. Allocated before
    // the matmul_plan_t/eng.configure() below since oc_left is passed to
    // the engine as an eager plan.avl (its live value doubles as the AVL).
    const x_block_t tmp = pool.new_int(3);
    const Reg oc_left = tmp[0];
    const Reg vin_ptr = tmp[1];
    const Reg dst_ptr = tmp[2];

    matmul_plan_t plan;
    plan.dti = to_rvjit_optype(dt);
    plan.dto = to_rvjit_optype(dt_c);
    plan.dtb = plan.dti;
    plan.ptra = vin_wptr;
    plan.ptrb = ptr_bcast;
    plan.ptrc = dst_wptr;
    plan.strides
            = matmul_strides_t::from_bytes(lda_bytes, ldb_bytes, ldc_bytes);
    plan.max_n_ur = jcp.ur;

    plan.n_loop = loop_t::unroll_and_switch().limit(
            [&](const Reg &r) { ld(r, args, GET_OFF(bcast_dim)); });
    plan.k_loop = loop_t::unroll().limit(
            [&](const Reg &r) { ld(r, args, GET_OFF(reduce_dim)); });
    plan.avl = oc_left;

    if (!eng.configure(plan)) {
        VERROR(primitive, create,
                "rv64: 1x1 conv: failed to configure "
                "matmul component (jcp.ur too large for "
                "the available register file)");
        return;
    }

    // Post-ops: the K-loop's tmp is dead once dense_loop completes, reuse it
    const Reg flag = eng.k_loop().tmp();
    const Reg Ktmp = eng.k_loop().tmp();
    // Bias pointer: read-only within a tile, still needs a per-OC-block advance
    const Reg bias_wptr = jcp.with_bias ? pool.new_int() : Reg();

    // Code start

    pool.preserve();

    // Load parameters
    ld(vin_wptr, args, GET_OFF(load_data));
    ld(ptr_bcast, args, GET_OFF(bcast_data));
    ld(dst_wptr, args, GET_OFF(output_data));
    ld(oc_left, args, GET_OFF(load_dim));
    if (jcp.with_bias) ld(bias_wptr, args, GET_OFF(bias_data));

    ca.init_constant(vin_oc_step);
    ca.init_constant(dst_oc_step);
    ca.init_constant(neg_oc_block_c);

    // OC loop
    cf.while_(branch_t::gtz(oc_left), [&] {
        // Save bases for the next-OC-block advance below
        mv(vin_ptr, vin_wptr);
        mv(dst_ptr, dst_wptr);

        eng.generate([&](v_block_t c, v_block_t vtmp) {
            const VReg t = vtmp(0);

            ld(flag, args, GET_OFF(first_last_flag));
            andi(flag, flag, FLAG_REDUCE_FIRST);
            cf.if_(branch_t::nez(flag), [&](bool is_first) {
                if (is_first) {
                    // First reduction: add bias (if any), write acc
                    if (jcp.with_bias) {
                        cf.if_(branch_t::nez(bias_wptr), [&] {
                            mem.vle(t, bias_wptr, to_rvjit_sew(dt_c));
                            for (int n = 0; n < c.size(); ++n)
                                vfadd_vv(c[n], c[n], t);
                        });
                    }
                    for (int n = 0; n < c.size(); ++n) {
                        mem.vse(c[n], dst_wptr, to_rvjit_sew(dt_c));
                        ca.add_const(dst_wptr, dst_wptr, eng.ldc());
                    }
                } else {
                    // Subsequent reduction: accumulate into C
                    for (int n = 0; n < c.size(); ++n) {
                        mem.vle(t, dst_wptr, to_rvjit_sew(dt_c));
                        vfadd_vv(t, t, c[n]);
                        mem.vse(t, dst_wptr, to_rvjit_sew(dt_c));
                        ca.add_const(dst_wptr, dst_wptr, eng.ldc());
                    }
                }
            });

            // Advance src pointer for next spatial tile; reuse Ktmp
            const const_t bcast_off
                    = ca.init_constant(const_t(c.size() * ldb_bytes, Ktmp));
            ca.add_const(ptr_bcast, ptr_bcast, bcast_off);
        });

        // Advance to the next OC block
        ld(ptr_bcast, args, GET_OFF(bcast_data));
        ca.add_const(oc_left, oc_left, neg_oc_block_c);
        ca.add_const(vin_wptr, vin_ptr, vin_oc_step);
        ca.add_const(dst_wptr, dst_ptr, dst_oc_step);
        if (jcp.with_bias) ca.add_const(bias_wptr, bias_wptr, dst_oc_step);
    });

    pool.restore();
    ret();
#else
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
