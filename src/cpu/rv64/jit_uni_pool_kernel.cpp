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

#include <cfloat>
#include <cstddef>
#include <cstring>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rv64/jit_uni_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pool_kernel_t<isa, d_type>::jit_uni_pool_kernel_t(
        const jit_pool_conf_t &jpp)
    : jit_generator_t(jpp.alg == alg_kind::pooling_max ? "jit_rvv_pool_fwd_max"
                      : jpp.alg == alg_kind::pooling_avg_include_padding
                      ? "jit_rvv_pool_fwd_avg_inc"
                      : "jit_rvv_pool_fwd_avg_exc")
    , jpp_(jpp)
    , is_max_pool_(jpp.alg == alg_kind::pooling_max) {
    create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pool_kernel_t<isa, d_type>::init_conf(
        jit_pool_conf_t &jpp, primitive_attr_t &attr, const pooling_pd_t *ppd) {
    using namespace alg_kind;
    using namespace format_tag;

    const memory_desc_wrapper src_d(ppd->src_md());
    const memory_desc_wrapper dst_d(ppd->dst_md());
    const int ndims = src_d.ndims();
    const auto &pd = *ppd->desc();

    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];
    jpp.id = (ndims == 5) ? src_d.dims()[ndims - 3] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[ndims - 3] : 1;
    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jpp.ow = dst_d.dims()[ndims - 1];

    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = (ndims == 3) ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = (ndims == 3) ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];
    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = (ndims == 3) ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    jpp.alg = pd.alg_kind;
    jpp.is_backward = false;
    jpp.src_dt = src_d.data_type();
    jpp.dst_dt = dst_d.data_type();
    jpp.dt_size = types::data_type_size(jpp.src_dt);
    jpp.isa = isa;
    jpp.nthr = dnnl_get_max_threads();
    // Output-width unroll for the shape-baked interior kernel. 4 keeps register
    // pressure low (ur_w accumulators + tmp + mask) while giving real input
    // reuse for the common small strides.
    jpp.ur_w = 4;

    // Memory layout: nspc (vectorize along C) or ncsp (vectorize along OW).
    // No blocked-format support — plain layouts are handled natively.
    const auto nspc_tag = utils::pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto ncsp_tag = utils::pick(ndims - 3, ncw, nchw, ncdhw);
    if (src_d.matches_tag(nspc_tag) && dst_d.matches_tag(nspc_tag))
        jpp.tag_kind = jit_pool_tag_kind_t::nspc;
    else if (src_d.matches_tag(ncsp_tag) && dst_d.matches_tag(ncsp_tag))
        jpp.tag_kind = jit_pool_tag_kind_t::ncsp;
    else
        return status::unimplemented;

    // Post-ops: the pd's post_ops_ok() already accepted only chains this kernel
    // can fuse in-kernel via the post-op injector (f32: any eltwise + at most one
    // binary; f16: eltwise-only); everything else went to ref_pooling. The
    // fuse_eltwise / fuse_binary flags below drive that injector fusion.
    // with_relu (f32 single-ReLU) is separate — it only feeds the empty-window
    // fill value (empty_window_value()), not the fusion.
    const auto &po = attr.post_ops_;
    jpp.post_ops = po;
    jpp.with_postops = !po.has_default_values();
    jpp.with_relu = false;
    jpp.fuse_eltwise = jpp.fuse_binary = false;
    jpp.relu_alpha = 0.f;
    if (po.len() == 1) {
        const auto &e = po.entry_[0];
        // with_relu is an f32-only flag feeding empty_window_value(); it is
        // not the fusion mechanism. f16 eltwise (incl ReLU) fuses via
        // fuse_eltwise in generate_f16.
        if (e.is_eltwise() && e.eltwise.alg == eltwise_relu
                && d_type == data_type::f32) {
            jpp.with_relu = true;
            jpp.relu_alpha = e.eltwise.alpha;
        }
    }
    // Post-op chains (any number of eltwise + at most one binary, in attribute
    // order) are fused in-kernel via the post-op injector. An eltwise-only chain
    // fuses for both f32 (generate_f32 / interior) and f16 (generate_f16, at
    // f32). A chain that contains the binary fuses for f32 only, via the
    // channel-vectorized single-position path (the driver forces it when
    // fuse_binary) so the rhs broadcast is uniform; the injector loads the rhs
    // per channel chunk. post_ops_ok already gated the chain shape/algs.
    const bool inj_ok
            = injector::jit_uni_postops_injector_t<isa>::post_ops_ok(po);
    bool po_has_binary = false;
    for (int i = 0; i < po.len(); i++)
        if (po.entry_[i].is_binary()) po_has_binary = true;
    jpp.fuse_eltwise = jpp.with_postops && inj_ok && !po_has_binary;
    jpp.fuse_binary = jpp.with_postops && inj_ok && po_has_binary
            && d_type == data_type::f32;
    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_kernel_t<isa, d_type>::generate() {
    if (d_type == data_type::f16)
        generate_f16();
    else
        generate_f32();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_kernel_t<isa, d_type>::generate_f32() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const VReg v_mask(0);
    const VReg v_acc(4), v_tmp(8);

    // Classify a fused binary post-op (if any) to pick the rhs load form. The
    // driver routes binary through the channel-vectorized single-position path,
    // so the lanes are channels: per-tensor -> scalar (flw); per-oc [1,C,1,..]
    // -> [C]-contiguous per-element (vle); full-dst -> per-element strided by the
    // dst channel stride (vlse), since src1 follows the dst layout.
    bool bin_scalar = false, bin_strided = false;
    if (jpp_.fuse_binary)
        for (int i = 0; i < jpp_.post_ops.len(); i++) {
            const auto &e = jpp_.post_ops.entry_[i];
            if (!e.is_binary()) continue;
            const memory_desc_wrapper s1(e.binary.src1_desc);
            bin_scalar = s1.nelems() == 1;
            const bool per_oc = !bin_scalar && s1.nelems() == (dim_t)jpp_.c;
            bin_strided = !bin_scalar && !per_oc;
        }

    // Save callee-saved regs (s0-s9) holding loop invariants across the nested
    // id/ih/iw/channel loops. f32 always processes a single output position
    // (the nspc interior heavy path uses the shape-baked kernel), so no
    // position loop / row-unrolling is emitted here — keeping per-call overhead
    // minimal for the high-call-count ncsp and boundary paths.
    // s10 additionally holds the binary post-op rhs pointer (advanced per
    // channel chunk) when a binary post-op is fused.
    const int stack_size = jpp_.fuse_binary ? 96 : 80;
    addi(sp, sp, -stack_size);
    sd(s0, sp, 0);
    sd(s1, sp, 8);
    sd(s2, sp, 16);
    sd(s3, sp, 24);
    sd(s4, sp, 32);
    sd(s5, sp, 40);
    sd(s6, sp, 48);
    sd(s7, sp, 56);
    sd(s8, sp, 64);
    sd(s9, sp, 72);
    if (jpp_.fuse_binary) sd(s10, sp, 80);

    // Load params from jit_uni_pooling_args_t
    using p_t = jit_uni_pooling_args_t;
    ld(s0, reg_param, static_cast<int>(offsetof(p_t, src)));
    ld(s1, reg_param, static_cast<int>(offsetof(p_t, dst)));
    ld(s2, reg_param, static_cast<int>(offsetof(p_t, channels)));
    ld(s6, reg_param, static_cast<int>(offsetof(p_t, id_start)));
    ld(s7, reg_param, static_cast<int>(offsetof(p_t, ih_start)));
    ld(a1, reg_param, static_cast<int>(offsetof(p_t, iw_start)));
    ld(t5, reg_param, static_cast<int>(offsetof(p_t, id_end)));
    ld(t6, reg_param, static_cast<int>(offsetof(p_t, ih_end)));
    ld(a2, reg_param, static_cast<int>(offsetof(p_t, iw_end)));
    ld(t2, reg_param, static_cast<int>(offsetof(p_t, inW_stride)));
    ld(t3, reg_param, static_cast<int>(offsetof(p_t, inD_stride)));
    ld(s3, reg_param, static_cast<int>(offsetof(p_t, w_spatial_byte_stride)));
    flw(fa0, reg_param, static_cast<int>(offsetof(p_t, init_val)));
    flw(ft1, reg_param, static_cast<int>(offsetof(p_t, scale_val)));

    fmv_w_x(fa1, x0); // f_zero = 0.0

    // Compute byte strides
    slli(s4, t2, 2); // inW_stride_bytes = inW_stride * 4
    slli(s5, t3, 2); // inD_stride_bytes = inD_stride * 4

    ld(s8, reg_param, static_cast<int>(offsetof(p_t, src_vec_byte_stride)));
    ld(s9, reg_param, static_cast<int>(offsetof(p_t, dst_vec_byte_stride)));
    if (jpp_.fuse_binary)
        ld(s10, reg_param, static_cast<int>(offsetof(p_t, post_op_rhs)));

    addi(t4, x0, 4); // constant for unit-stride comparison

    // Channel loop: process channels in vector chunks (single output position).
    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s2, ch_done);

    // Set vector length for remaining channels
    vsetvli(t0, s2, SEW::e32, LMUL::m1);

    // Initialize accumulator
    if (is_max_pool_) {
        // init_val = -FLT_MAX for max pool
        vfmv_v_f(v_acc, fa0);
    } else {
        vfmv_v_f(v_acc, fa1); // zero
    }

    // Depth (id) loop
    mv(a3, s6); // reg_id = id_start
    mul(t1, a3, s5);
    add(t1, s0, t1); // t1 = depth_ptr = src + id * inD_stride_bytes

    Label id_loop, id_done;
    L(id_loop);
    bge(a3, t5, id_done); // if id >= id_end, done

    // Height (ih) loop
    mv(a4, s7); // reg_ih = ih_start
    mul(t2, a4, s4);
    add(a6, t1, t2); // a6 = row_ptr

    Label ih_loop, ih_done;
    L(ih_loop);
    bge(a4, t6, ih_done); // if ih >= ih_end, done

    // Width (iw) loop
    mv(a5, a1); // reg_iw = iw_start
    mul(t2, a5, s3);
    add(a7, a6, t2); // a7 = src_ptr

    Label iw_loop, iw_done;
    L(iw_loop);
    bge(a5, a2, iw_done); // if iw >= iw_end, done

    // Load — use vle32_v for unit stride to avoid potential uarch penalty
    {
        Label strided_src, src_ld_done;
        bne(s8, t4, strided_src);
        vle32_v(v_tmp, a7);
        j_(src_ld_done);
        L(strided_src);
        vlse32_v(v_tmp, a7, s8);
        L(src_ld_done);
    }
    if (is_max_pool_) {
        vmflt_vv(v_mask, v_acc, v_tmp);
        vmerge_vvm(v_acc, v_acc, v_tmp);
    } else {
        vfadd_vv(v_acc, v_acc, v_tmp);
    }

    addi(a5, a5, 1); // iw++
    add(a7, a7, s3); // advance src_ptr by w_spatial_byte_stride
    j_(iw_loop);
    L(iw_done);

    addi(a4, a4, 1); // ih++
    add(a6, a6, s4); // advance row_ptr by inW_stride_bytes
    j_(ih_loop);
    L(ih_done);

    addi(a3, a3, 1); // id++
    add(t1, t1, s5); // advance depth_ptr by inD_stride_bytes
    j_(id_loop);
    L(id_done);

    // Apply avg pooling divide
    if (!is_max_pool_) { vfmul_vf(v_acc, v_acc, ft1); }

    // Apply the fused post-op chain (any number of eltwise + at most one binary,
    // in attribute order) via the in-kernel injector. Eltwise covers ReLU and
    // the other supported algs; binary reads its rhs from s10 (host-positioned,
    // advanced per channel chunk below). This is the f32 path; the f16 kernel
    // fuses its eltwise chain in generate_f16 (f16 + binary -> ref_pooling).
    if (jpp_.fuse_eltwise || jpp_.fuse_binary) {
        eltwise_injector::static_params_t esp(
                VReg(12), VReg(16), VReg(20), fa3, fa4, t2, /*is_fwd=*/true);
        // Binary rhs scratch v24/fa5; full-dst uses a strided (vlse) load keyed
        // on the dst channel stride (s9), per-oc/scalar the contiguous form.
        binary_injector::static_params_t bsp_contig(VReg(24), fa5, s10);
        binary_injector::static_params_t bsp_strided(VReg(24), fa5, s10, s9);
        injector::jit_uni_postops_injector_t<isa> po_inj(this, jpp_.post_ops,
                esp,
                jpp_.fuse_binary ? (bin_strided ? &bsp_strided : &bsp_contig)
                                 : nullptr);
        po_inj.compute_vector(v_acc.getIdx());
    }
    // Store result — use vse32_v for unit stride
    {
        Label strided_dst, dst_st_done;
        bne(s9, t4, strided_dst);
        vse32_v(v_acc, s1);
        j_(dst_st_done);
        L(strided_dst);
        vsse32_v(v_acc, s1, s9);
        L(dst_st_done);
    }

    // Advance src/dst pointers by vl * stride
    {
        Label strided_src_adv, src_adv_done;
        bne(s8, t4, strided_src_adv);
        slli(t1, t0, 2);
        j_(src_adv_done);
        L(strided_src_adv);
        mul(t1, t0, s8);
        L(src_adv_done);
    }
    add(s0, s0, t1);
    {
        Label strided_dst_adv, dst_adv_done;
        bne(s9, t4, strided_dst_adv);
        slli(t1, t0, 2);
        j_(dst_adv_done);
        L(strided_dst_adv);
        mul(t1, t0, s9);
        L(dst_adv_done);
    }
    add(s1, s1, t1);
    if (jpp_.fuse_binary && !bin_scalar) {
        // advance the rhs by vl * per-channel byte stride (full-dst: dst channel
        // stride s9; per-oc: contiguous element size).
        if (bin_strided)
            mul(t1, t0, s9);
        else
            slli(t1, t0, 2);
        add(s10, s10, t1);
    }
    sub(s2, s2, t0); // channels -= vl

    j_(ch_loop);
    L(ch_done);

    // Restore callee-saved regs
    ld(s0, sp, 0);
    ld(s1, sp, 8);
    ld(s2, sp, 16);
    ld(s3, sp, 24);
    ld(s4, sp, 32);
    ld(s5, sp, 40);
    ld(s6, sp, 48);
    ld(s7, sp, 56);
    ld(s8, sp, 64);
    ld(s9, sp, 72);
    if (jpp_.fuse_binary) ld(s10, sp, 80);
    addi(sp, sp, stack_size);

    ret();
#else
    ret();
#endif
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_kernel_t<isa, d_type>::generate_f16() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const VReg v_mask(0);
    // max: v_acc(f16m1)=v4, v_tmp(f16m1)=v8.
    // avg: v_acc(f32m2)=v4-v5, v_tmp(f16m1)=v8 (load buffer + narrowed result).
    const VReg v_acc(4), v_tmp(8);
    const VReg v_res = is_max_pool_ ? v_acc : v_tmp;

    const int stack_size = 112;
    addi(sp, sp, -stack_size);
    sd(s0, sp, 0);
    sd(s1, sp, 8);
    sd(s2, sp, 16);
    sd(s3, sp, 24);
    sd(s4, sp, 32);
    sd(s5, sp, 40);
    sd(s6, sp, 48);
    sd(s7, sp, 56);
    sd(s8, sp, 64);
    sd(s9, sp, 72);
    sd(s10, sp, 80);
    sd(s11, sp, 88);

    using p_t = jit_uni_pooling_args_t;
    ld(s0, reg_param, static_cast<int>(offsetof(p_t, src)));
    ld(s1, reg_param, static_cast<int>(offsetof(p_t, dst)));
    ld(s2, reg_param, static_cast<int>(offsetof(p_t, channels)));
    ld(s6, reg_param, static_cast<int>(offsetof(p_t, id_start)));
    ld(s7, reg_param, static_cast<int>(offsetof(p_t, ih_start)));
    ld(a1, reg_param, static_cast<int>(offsetof(p_t, iw_start)));
    ld(t5, reg_param, static_cast<int>(offsetof(p_t, id_end)));
    ld(t6, reg_param, static_cast<int>(offsetof(p_t, ih_end)));
    ld(a2, reg_param, static_cast<int>(offsetof(p_t, iw_end)));
    ld(t2, reg_param, static_cast<int>(offsetof(p_t, inW_stride)));
    ld(t3, reg_param, static_cast<int>(offsetof(p_t, inD_stride)));
    ld(s3, reg_param, static_cast<int>(offsetof(p_t, w_spatial_byte_stride)));
    flw(ft1, reg_param, static_cast<int>(offsetof(p_t, scale_val)));

    // f16 element strides: inW_stride / inD_stride are element counts -> * 2.
    slli(s4, t2, 1);
    slli(s5, t3, 1);

    ld(s8, reg_param, static_cast<int>(offsetof(p_t, src_vec_byte_stride)));
    ld(s9, reg_param, static_cast<int>(offsetof(p_t, dst_vec_byte_stride)));

    addi(t4, x0, 2); // unit-stride comparison constant (f16 = 2 bytes)

    mv(s10, s0);
    mv(s11, s1);
    ld(t0, reg_param, static_cast<int>(offsetof(p_t, n_pos)));
    sd(t0, sp, 96);

    // In-kernel eltwise post-op injector (f16 computes the chain at f32). t1 is
    // a scratch GPR reused across window iterations, free at the inject point.
    post_ops_t no_po;
    eltwise_injector::static_params_t esp(
            VReg(12), VReg(16), VReg(20), fa3, fa4, t1, /*is_fwd=*/true);
    injector::jit_uni_postops_injector_t<isa> po_inj(
            this, jpp_.fuse_eltwise ? jpp_.post_ops : no_po, esp);

    Label pos_loop, pos_done;
    L(pos_loop);
    ld(t0, sp, 96);
    beqz(t0, pos_done);

    mv(s0, s10);
    mv(s1, s11);
    ld(s2, reg_param, static_cast<int>(offsetof(p_t, channels)));

    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s2, ch_done);

    // Initialize accumulator. Use vmv.v.x with the raw bit pattern to avoid
    // f16 scalar NaN-boxing concerns.
    if (is_max_pool_) {
        vsetvli(t0, s2, SEW::e16, LMUL::m1);
        li(t1, 0xFBFF); // f16 lowest (-65504.0)
        vmv_v_x(v_acc, t1);
    } else {
        // avg: f32m2 accumulator zeroed; window runs under e16/m1 (same vl).
        vsetvli(t0, s2, SEW::e32, LMUL::m2);
        vmv_v_x(v_acc, x0);
        vsetvli(t0, s2, SEW::e16, LMUL::m1);
    }

    mv(a3, s6);
    mul(t1, a3, s5);
    add(t1, s0, t1);
    Label id_loop, id_done;
    L(id_loop);
    bge(a3, t5, id_done);

    mv(a4, s7);
    mul(t2, a4, s4);
    add(a6, t1, t2);
    Label ih_loop, ih_done;
    L(ih_loop);
    bge(a4, t6, ih_done);

    mv(a5, a1);
    mul(t2, a5, s3);
    add(a7, a6, t2);
    Label iw_loop, iw_done;
    L(iw_loop);
    bge(a5, a2, iw_done);

    {
        Label strided_src, src_ld_done;
        bne(s8, t4, strided_src);
        vle16_v(v_tmp, a7);
        j_(src_ld_done);
        L(strided_src);
        vlse16_v(v_tmp, a7, s8);
        L(src_ld_done);
    }
    if (is_max_pool_) {
        vmflt_vv(v_mask, v_acc, v_tmp);
        vmerge_vvm(v_acc, v_acc, v_tmp);
    } else {
        // f32m2 += widen(f16m1), evaluated under the e16 vtype.
        vfwadd_wv(v_acc, v_acc, v_tmp);
    }

    addi(a5, a5, 1);
    add(a7, a7, s3);
    j_(iw_loop);
    L(iw_done);

    addi(a4, a4, 1);
    add(a6, a6, s4);
    j_(ih_loop);
    L(ih_done);

    addi(a3, a3, 1);
    add(t1, t1, s5);
    j_(id_loop);
    L(id_done);

    // avg: scale (e32/m2), apply the fused eltwise post-op (at f32), then narrow
    // to f16 (e16/m1).
    if (!is_max_pool_) {
        vsetvli(t0, s2, SEW::e32, LMUL::m2);
        vfmul_vf(v_acc, v_acc, ft1);
        if (jpp_.fuse_eltwise) po_inj.compute_vector(v_acc.getIdx());
        vsetvli(t0, s2, SEW::e16, LMUL::m1);
        vfncvt_f_f_w(v_tmp, v_acc);
    } else if (jpp_.fuse_eltwise) {
        // max: the accumulator is f16; widen to f32 (v24/m2), apply the eltwise
        // injector, then narrow back in place so the store path is unchanged.
        vsetvli(t0, s2, SEW::e16, LMUL::m1);
        vfwcvt_f_f_v(VReg(24), v_acc);
        vsetvli(t0, s2, SEW::e32, LMUL::m2);
        po_inj.compute_vector(24);
        vsetvli(t0, s2, SEW::e16, LMUL::m1);
        vfncvt_f_f_w(v_acc, VReg(24));
    }

    {
        Label strided_dst, dst_st_done;
        bne(s9, t4, strided_dst);
        vse16_v(v_res, s1);
        j_(dst_st_done);
        L(strided_dst);
        vsse16_v(v_res, s1, s9);
        L(dst_st_done);
    }

    // Advance src/dst by vl * stride (f16 unit stride = vl * 2).
    {
        Label strided_src_adv, src_adv_done;
        bne(s8, t4, strided_src_adv);
        slli(t1, t0, 1);
        j_(src_adv_done);
        L(strided_src_adv);
        mul(t1, t0, s8);
        L(src_adv_done);
    }
    add(s0, s0, t1);
    {
        Label strided_dst_adv, dst_adv_done;
        bne(s9, t4, strided_dst_adv);
        slli(t1, t0, 1);
        j_(dst_adv_done);
        L(strided_dst_adv);
        mul(t1, t0, s9);
        L(dst_adv_done);
    }
    add(s1, s1, t1);
    sub(s2, s2, t0);

    j_(ch_loop);
    L(ch_done);

    ld(t0, reg_param, static_cast<int>(offsetof(p_t, pos_src_byte_stride)));
    add(s10, s10, t0);
    ld(t0, reg_param, static_cast<int>(offsetof(p_t, pos_dst_byte_stride)));
    add(s11, s11, t0);
    ld(t0, sp, 96);
    addi(t0, t0, -1);
    sd(t0, sp, 96);
    j_(pos_loop);
    L(pos_done);

    ld(s0, sp, 0);
    ld(s1, sp, 8);
    ld(s2, sp, 16);
    ld(s3, sp, 24);
    ld(s4, sp, 32);
    ld(s5, sp, 40);
    ld(s6, sp, 48);
    ld(s7, sp, 56);
    ld(s8, sp, 64);
    ld(s9, sp, 72);
    ld(s10, sp, 80);
    ld(s11, sp, 88);
    addi(sp, sp, stack_size);
    ret();
#else
    ret();
#endif
}

template struct jit_uni_pool_kernel_t<v, data_type::f32>;
template struct jit_uni_pool_kernel_t<zvfh, data_type::f16>;

// === Shape-baked interior kernel (nspc, ur_w input reuse) ===

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pool_interior_kernel_t<isa, d_type>::jit_uni_pool_interior_kernel_t(
        const jit_pool_conf_t &ajpp)
    : jit_generator_t("jit_rvv_pool_interior"), jpp_(ajpp) {
    create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_interior_kernel_t<isa, d_type>::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    generate_nspc();
#else
    ret();
#endif
}

#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pool_interior_kernel_t<isa, d_type>::generate_nspc() {
    using p_t = jit_uni_pool_interior_args_t;
    auto fbits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };

    const bool is_max = jpp_.alg == alg_kind::pooling_max;
    const VReg v_mask(0);
    const bool is_avg_exclude
            = jpp_.alg == alg_kind::pooling_avg_exclude_padding;
    const int kw = jpp_.kw;
    const int sw = jpp_.stride_w;
    const int ur_w = jpp_.ur_w;
    const int max_p = (ur_w - 1) * sw + kw; // W positions in the unrolled sweep
    const float init_val = is_max ? -FLT_MAX : 0.0f;
    const float avg_inc_scale = 1.0f / (float)(jpp_.kd * jpp_.kh * jpp_.kw);

    const Reg reg_param = a0;
    const VReg v_tmp(4 + ur_w);
    auto acc = [&](int j) { return VReg(4 + j); };

    // Prologue: save s0-s11.
    const int stack = 96;
    addi(sp, sp, -stack);
    sd(s0, sp, 0);
    sd(s1, sp, 8);
    sd(s2, sp, 16);
    sd(s3, sp, 24);
    sd(s4, sp, 32);
    sd(s5, sp, 40);
    sd(s6, sp, 48);
    sd(s7, sp, 56);
    sd(s8, sp, 64);
    sd(s9, sp, 72);
    sd(s10, sp, 80);
    sd(s11, sp, 88);

    // Persistent state (survives the block/channel/window loops).
    ld(s0, reg_param, (int)offsetof(p_t, src)); // block_base_src
    ld(s1, reg_param, (int)offsetof(p_t, dst)); // block_base_dst
    ld(s2, reg_param, (int)offsetof(p_t, channels)); // channels_orig
    ld(s3, reg_param, (int)offsetof(p_t, kh_count));
    ld(s4, reg_param, (int)offsetof(p_t, kd_count));
    ld(s5, reg_param, (int)offsetof(p_t, w_stride));
    ld(s6, reg_param, (int)offsetof(p_t, inW_stride));
    ld(s7, reg_param, (int)offsetof(p_t, inD_stride));
    ld(s10, reg_param, (int)offsetof(p_t, n_blocks));
    // Per-block base strides derived from w_stride: src advances ur_w*sw W
    // positions, dst advances ur_w output columns (dst column stride == w_stride
    // for nspc).
    li(t0, (uint32_t)(ur_w * sw));
    mul(s8, s5, t0);
    li(t0, (uint32_t)ur_w);
    mul(s9, s5, t0);

    // FP constants.
    li(t0, fbits(init_val));
    fmv_w_x(fa0, t0);
    fmv_w_x(fa3, x0); // zero
    if (is_avg_exclude) {
        flw(fa1, reg_param, (int)offsetof(p_t, scale_val));
    } else if (!is_max) { // avg_include: scale is baked
        li(t0, fbits(avg_inc_scale));
        fmv_w_x(fa1, t0);
    }
    // In-kernel eltwise post-op injector for the shape-baked f32 nspc interior
    // kernel: the eltwise chain is applied when fuse_eltwise, else no_po keeps
    // the injector empty. A fused binary makes the driver skip this interior
    // kernel (n_blocks=0; every column goes through the channel-vec path), so
    // only eltwise chains reach here.
    post_ops_t no_po;
    eltwise_injector::static_params_t esp(
            VReg(12), VReg(16), VReg(20), fa4, fa5, t6, /*is_fwd=*/true);
    injector::jit_uni_postops_injector_t<isa> po_inj(
            this, jpp_.fuse_eltwise ? jpp_.post_ops : no_po, esp);

    Label block_loop, block_done;
    L(block_loop);
    beqz(s10, block_done);

    mv(a4, s0); // working_src_base
    mv(a5, s1); // working_dst_base
    mv(a6, s2); // working channels

    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(a6, ch_done);
    vsetvli(t0, a6, SEW::e32, LMUL::m1); // vl in t0

    for (int j = 0; j < ur_w; j++)
        vfmv_v_f(acc(j), fa0);

    // Depth (kd) loop — runtime count.
    mv(t1, a4); // id_ptr
    mv(t2, s4); // id_cnt
    Label id_loop, id_done;
    L(id_loop);
    beqz(t2, id_done);

    // Height (kh) loop — runtime count.
    mv(t3, t1); // ih_ptr
    mv(t4, s3); // ih_cnt
    Label ih_loop, ih_done;
    L(ih_loop);
    beqz(t4, ih_done);

    // Width sweep — fully unrolled; each loaded input feeds every output column
    // whose window covers it (input reuse).
    mv(t5, t3); // w_ptr
    for (int p = 0; p < max_p; p++) {
        vle32_v(v_tmp, t5);
        for (int j = 0; j < ur_w; j++) {
            if (j * sw <= p && p < j * sw + kw) {
                if (is_max) {
                    vmflt_vv(v_mask, acc(j), v_tmp);
                    vmerge_vvm(acc(j), acc(j), v_tmp);
                } else
                    vfadd_vv(acc(j), acc(j), v_tmp);
            }
        }
        if (p + 1 < max_p) add(t5, t5, s5);
    }

    addi(t4, t4, -1);
    add(t3, t3, s6); // ih_ptr += inW_stride
    j_(ih_loop);
    L(ih_done);

    addi(t2, t2, -1);
    add(t1, t1, s7); // id_ptr += inD_stride
    j_(id_loop);
    L(id_done);

    // Scale (avg) + fused eltwise post-op chain + store for each of the ur_w
    // output columns.
    mv(a1, a5); // dst_ptr
    for (int j = 0; j < ur_w; j++) {
        if (!is_max) vfmul_vf(acc(j), acc(j), fa1);
        if (jpp_.fuse_eltwise) po_inj.compute_vector(acc(j).getIdx());
        vse32_v(acc(j), a1);
        if (j + 1 < ur_w) add(a1, a1, s5); // dst column stride == w_stride
    }

    // Advance to the next channel chunk.
    slli(a2, t0, 2); // vl * sizeof(float)
    add(a4, a4, a2);
    add(a5, a5, a2);
    sub(a6, a6, t0);
    j_(ch_loop);
    L(ch_done);

    // Advance to the next ur_w output block.
    add(s0, s0, s8);
    add(s1, s1, s9);
    addi(s10, s10, -1);
    j_(block_loop);
    L(block_done);

    // Epilogue.
    ld(s0, sp, 0);
    ld(s1, sp, 8);
    ld(s2, sp, 16);
    ld(s3, sp, 24);
    ld(s4, sp, 32);
    ld(s5, sp, 40);
    ld(s6, sp, 48);
    ld(s7, sp, 56);
    ld(s8, sp, 64);
    ld(s9, sp, 72);
    ld(s10, sp, 80);
    ld(s11, sp, 88);
    addi(sp, sp, stack);
    ret();
}
#endif

template struct jit_uni_pool_interior_kernel_t<v, data_type::f32>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
