/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2023 KNS Group LLC (YADRO)
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

#include <algorithm>
#include <cfloat>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/jit_uni_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

template <cpu_isa_t isa, data_type_t d_type>
using kernel_t = jit_uni_pool_kernel_t<isa, d_type>;

struct init_scale_t {
    float init_val;
    float scale_val;
};

// Initial accumulator value and post-window scale for a vectorized output.
init_scale_t compute_init_scale(alg_kind_t alg, int id_start, int id_end,
        int ih_start, int ih_end, int iw_start, int iw_end, dim_t kerD,
        dim_t kerH, dim_t kerW) {
    if (alg == alg_kind::pooling_max) {
        return {-FLT_MAX, 0.0f};
    } else if (alg == alg_kind::pooling_avg_include_padding) {
        return {0.0f, 1.0f / (float)(kerD * kerH * kerW)};
    } else {
        const int count = std::max(0, id_end - id_start)
                * std::max(0, ih_end - ih_start)
                * std::max(0, iw_end - iw_start);
        return {0.0f, (count > 0) ? 1.0f / (float)count : 0.0f};
    }
}

// Value the kernel would write for an output whose pooling window lies entirely
// in padding: the accumulator init (max -> dtype lowest, avg -> 0), which avg
// scaling leaves at 0, with the fused f32 single-ReLU baked in (with_relu). Used
// to fast-fill such outputs, skipping the per-output JIT call. Post-ops are fused
// in-kernel now (there is no separate post-op pass); a fused chain this value
// cannot reproduce has its empty windows routed through the kernel instead (see
// execute_forward), and unsupported chains fall back to ref_pooling.
template <typename data_t>
data_t empty_window_value(const jit_pool_conf_t &jpp) {
    if (jpp.alg == alg_kind::pooling_max) {
        float v = (float)nstl::numeric_limits<data_t>::lowest();
        if (jpp.with_relu) { // fused only for f32
            if (jpp.relu_alpha == 0.0f)
                v = v < 0.0f ? 0.0f : v;
            else if (v <= 0.0f)
                v *= jpp.relu_alpha;
        }
        return (data_t)v;
    }
    return (data_t)0;
}

// Per-output-pixel kernel arguments for the nspc layout (vectorize along C).
template <typename data_t>
jit_uni_pooling_args_t make_pooling_params(alg_kind_t alg,
        const data_t *src_mb_base, data_t *dst, dim_t mb, dim_t channels,
        dim_t od, dim_t oh, dim_t ow, dim_t outD, dim_t outH, dim_t outW,
        dim_t inD, dim_t inH, dim_t inW, dim_t kerD, dim_t kerH, dim_t kerW,
        dim_t strideD, dim_t strideH, dim_t strideW, dim_t padFront,
        dim_t padTop, dim_t padLeft, dim_t inW_stride, dim_t inD_stride,
        bool with_relu, float relu_alpha) {

    const size_t dst_spatial_base
            = ((((size_t)mb * outD + od) * outH + oh) * outW + ow) * channels;

    const int od_offset = (int)(od * strideD - padFront);
    const int oh_offset = (int)(oh * strideH - padTop);
    const int ow_offset = (int)(ow * strideW - padLeft);

    const int id_start = std::max(od_offset, 0);
    const int ih_start = std::max(oh_offset, 0);
    const int iw_start = std::max(ow_offset, 0);
    const int id_end = std::min(od_offset + (int)kerD, (int)inD);
    const int ih_end = std::min(oh_offset + (int)kerH, (int)inH);
    const int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

    auto iv = compute_init_scale(alg, id_start, id_end, ih_start, ih_end,
            iw_start, iw_end, kerD, kerH, kerW);

    jit_uni_pooling_args_t p;
    p.src = src_mb_base;
    p.dst = &dst[dst_spatial_base];
    p.channels = channels;
    p.id_start = id_start;
    p.ih_start = ih_start;
    p.iw_start = iw_start;
    p.id_end = id_end;
    p.ih_end = ih_end;
    p.iw_end = iw_end;
    p.inW_stride = inW_stride;
    p.inD_stride = inD_stride;
    p.w_spatial_byte_stride = channels * sizeof(data_t);
    p.init_val = iv.init_val;
    p.scale_val = iv.scale_val;
    p.relu_alpha = relu_alpha;
    p.with_relu = with_relu;
    p.src_vec_byte_stride = (dim_t)sizeof(data_t);
    p.dst_vec_byte_stride = (dim_t)sizeof(data_t);
    return p;
}

// Binary post-op rhs base for a channel-vectorized kernel call writing to
// p_dst. A full-dst src1 follows the dst layout, so it offsets with the output
// position (rhs + (p_dst - dst)); per-oc ([1,C,1,..]) and per-tensor src1 use
// the base directly. Returns nullptr when no binary post-op is fused.
template <typename data_t>
const void *binary_rhs_for(const jit_pool_conf_t &jpp, const void *rhs,
        const data_t *dst, const void *p_dst) {
    if (!jpp.fuse_binary || rhs == nullptr) return nullptr;
    bool full = false;
    for (int i = 0; i < jpp.post_ops.len(); i++) {
        const auto &e = jpp.post_ops.entry_[i];
        if (!e.is_binary()) continue;
        const memory_desc_wrapper s1(e.binary.src1_desc);
        full = s1.nelems() != 1 && s1.nelems() != (dim_t)jpp.c;
    }
    if (!full) return rhs;
    return static_cast<const void *>(static_cast<const data_t *>(rhs)
            + (static_cast<const data_t *>(p_dst) - dst));
}

// nspc: vectorize along C. Interior columns are handled by the shape-baked
// ur_w-reuse kernel (full ur_w blocks); the few boundary columns and the
// sub-ur_w interior remainder fall back to one vectorize-C call per column.
template <cpu_isa_t isa, data_type_t d_type>
void pooling_nspc(const kernel_t<isa, d_type> &kernel,
        const jit_uni_pool_interior_kernel_t<isa, d_type> &ikernel,
        const jit_pool_conf_t &jpp, const float *src, float *dst,
        const void *po_rhs) {
    const dim_t batch = jpp.mb, channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padFront = jpp.f_pad, padTop = jpp.t_pad, padLeft = jpp.l_pad;
    const alg_kind_t alg = jpp.alg;
    const bool with_relu = jpp.with_relu;
    const float relu_alpha = jpp.relu_alpha;
    const dim_t ur_w = jpp.ur_w;

    const dim_t inW_stride = inW * channels;
    const dim_t inD_stride = inH * inW * channels;

    const dim_t int_lo = std::max((padLeft + strideW - 1) / strideW, (dim_t)0);
    // Guard against truncating division: when inW + padLeft < kerW (kernel
    // wider than the padded input) the numerator is negative and integer
    // division would round toward zero, over-counting int_hi and marking a
    // partial window as full. In that case no full-window column exists.
    const dim_t int_hi = (inW + padLeft >= kerW)
            ? std::min(std::max((inW - kerW + padLeft) / strideW + 1, int_lo),
                      outW)
            : int_lo;

    parallel_nd(batch, outD, outH, [&](dim_t mb, dim_t od, dim_t oh) {
        const size_t mb_offset = (size_t)mb * (size_t)inD * inD_stride;
        const float *src_mb_base = &src[mb_offset];

        auto call_one = [&](dim_t ow) {
            auto p = make_pooling_params(alg, src_mb_base, dst, mb, channels,
                    od, oh, ow, outD, outH, outW, inD, inH, inW, kerD, kerH,
                    kerW, strideD, strideH, strideW, padFront, padTop, padLeft,
                    inW_stride, inD_stride, with_relu, relu_alpha);
            if ((p.id_start >= p.id_end || p.ih_start >= p.ih_end
                        || p.iw_start >= p.iw_end)
                    && !jpp.fuse_eltwise && !jpp.fuse_binary) {
                // Empty window, no fused post-op: fill channels, skip the kernel.
                const float ev = empty_window_value<float>(jpp);
                float *d = static_cast<float *>(p.dst);
                for (dim_t c = 0; c < channels; c++)
                    d[c] = ev;
                return;
            }
            p.post_op_rhs = binary_rhs_for(jpp, po_rhs, dst, p.dst);
            kernel(&p);
        };

        // D/H window for this (od, oh), clamped to the input. The shape-baked
        // kernel uses count-based loops (beqz + decrement), so it must not run
        // for a row whose window lies entirely in D/H padding (zero/negative
        // counts) — those rows fall through to the per-column path, which
        // clamps the window in the agnostic kernel (bge bounds).
        const int od_off = (int)(od * strideD - padFront);
        const int oh_off = (int)(oh * strideH - padTop);
        const int id_start = std::max(od_off, 0);
        const int ih_start = std::max(oh_off, 0);
        const int id_end = std::min(od_off + (int)kerD, (int)inD);
        const int ih_end = std::min(oh_off + (int)kerH, (int)inH);
        const dim_t kd_count = std::max(0, id_end - id_start);
        const dim_t kh_count = std::max(0, ih_end - ih_start);
        const bool row_in_pad = (kd_count == 0 || kh_count == 0);

        const dim_t n_int = (int_lo < int_hi) ? (int_hi - int_lo) : 0;
        // Binary post-ops are applied only on the channel-vectorized
        // single-position path, so skip the shape-baked interior kernel then.
        const dim_t n_blocks
                = (row_in_pad || jpp.fuse_binary) ? 0 : (n_int / ur_w);
        const dim_t baked_hi = int_lo + n_blocks * ur_w;

        // Left boundary columns.
        const dim_t lo = std::min(int_lo, outW);
        for (dim_t ow = 0; ow < lo; ++ow)
            call_one(ow);

        // Interior: full ur_w blocks via the shape-baked kernel. Reached only
        // when kd_count > 0 && kh_count > 0 (n_blocks gated on !row_in_pad).
        if (n_blocks > 0) {
            const int iw_start = (int)(int_lo * strideW - padLeft);

            jit_uni_pool_interior_args_t ip;
            ip.src = src_mb_base + (size_t)id_start * inD_stride
                    + (size_t)ih_start * inW_stride
                    + (size_t)iw_start * channels;
            ip.dst = dst
                    + (((((size_t)mb * outD + od) * outH + oh) * outW + int_lo)
                            * channels);
            ip.channels = channels;
            ip.kh_count = kh_count;
            ip.kd_count = kd_count;
            ip.n_blocks = n_blocks;
            ip.w_stride = (dim_t)channels * sizeof(float);
            ip.inW_stride = inW_stride * (dim_t)sizeof(float);
            ip.inD_stride = inD_stride * (dim_t)sizeof(float);
            const dim_t cnt = kd_count * kh_count * (dim_t)kerW;
            ip.scale_val = (cnt > 0) ? 1.0f / (float)cnt : 0.0f;
            ikernel(&ip);
        }

        // Interior remainder (< ur_w, or the whole interior when row_in_pad),
        // then right boundary columns.
        for (dim_t ow = baked_hi; ow < int_hi; ++ow)
            call_one(ow);
        for (dim_t ow = int_hi; ow < outW; ++ow)
            call_one(ow);
    });
}

// Vectorize along C (strided) for a single ncsp output column (mb, od, oh, ow).
// Used for OW==1 and for the boundary columns of the OW>1 path, where each
// column has its own (possibly truncated) window and so cannot join the
// interior row-unrolled call.
template <cpu_isa_t isa, data_type_t d_type>
void pool_column_vec_c(const kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, dim_t mb, dim_t od, dim_t oh,
        dim_t ow, const void *po_rhs) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padF = jpp.f_pad, padT = jpp.t_pad, padL = jpp.l_pad;

    const size_t spatial_size = (size_t)inD * inH * inW;
    const size_t dst_spatial_size = (size_t)outD * outH * outW;

    const int id_start = std::max((int)(od * strideD - padF), 0);
    const int id_end = std::min((int)(od * strideD - padF + kerD), (int)inD);
    const int ih_start = std::max((int)(oh * strideH - padT), 0);
    const int ih_end = std::min((int)(oh * strideH - padT + kerH), (int)inH);
    const int iw_start = std::max((int)(ow * strideW - padL), 0);
    const int iw_end = std::min((int)(ow * strideW - padL + kerW), (int)inW);

    const dim_t kd_cnt = std::max(0, id_end - id_start);
    const dim_t kh_cnt = std::max(0, ih_end - ih_start);
    const dim_t kw_cnt = std::max(0, iw_end - iw_start);

    data_t *dst_base = dst + (size_t)mb * channels * dst_spatial_size
            + od * outH * outW + oh * outW + ow;

    // Output whose window lies entirely in padding (legal for max /
    // avg_include; only avg_exclude is rejected upstream). With no fused post-op
    // we fill all channels directly with the value the kernel would produce and
    // skip the JIT call (avoids per-call overhead and an out-of-range src_base).
    // With a fused post-op the kernel must run so the post-op is applied to the
    // init value; the window loops are skipped (id_end==0, ...) so src is unread.
    const bool empty = (kd_cnt == 0 || kh_cnt == 0 || kw_cnt == 0);
    if (empty && !jpp.fuse_eltwise && !jpp.fuse_binary) {
        const data_t ev = empty_window_value<data_t>(jpp);
        for (dim_t c = 0; c < channels; c++)
            dst_base[(size_t)c * dst_spatial_size] = ev;
        return;
    }

    auto iv = compute_init_scale(jpp.alg, id_start, id_end, ih_start, ih_end,
            iw_start, iw_end, kerD, kerH, kerW);

    // Empty window: the kernel reads no src, so use the channel base to avoid
    // forming an out-of-range pointer (start indices can exceed the input).
    const data_t *src_base = empty ? src + (size_t)mb * channels * spatial_size
                                   : src + (size_t)mb * channels * spatial_size
                    + (size_t)id_start * inH * inW + (size_t)ih_start * inW
                    + iw_start;

    jit_uni_pooling_args_t p;
    p.src = src_base;
    p.dst = dst_base;
    p.channels = channels;
    p.id_start = 0;
    p.ih_start = 0;
    p.iw_start = 0;
    p.id_end = kd_cnt;
    p.ih_end = kh_cnt;
    p.iw_end = kw_cnt;
    p.inW_stride = inW;
    p.inD_stride = inH * inW;
    p.w_spatial_byte_stride = (dim_t)sizeof(data_t);
    p.init_val = iv.init_val;
    p.scale_val = iv.scale_val;
    p.relu_alpha = jpp.relu_alpha;
    p.with_relu = jpp.with_relu;
    p.src_vec_byte_stride = (dim_t)(spatial_size * sizeof(data_t));
    p.dst_vec_byte_stride = (dim_t)(dst_spatial_size * sizeof(data_t));
    p.post_op_rhs = binary_rhs_for(jpp, po_rhs, dst, p.dst);
    kernel(&p);
}

// ncsp, OW > 1: interior columns vectorize along OW (one row-unrolled call per
// channel); boundary columns vectorize along C (one call per column, all
// channels) via pool_column_vec_c.
template <cpu_isa_t isa, data_type_t d_type>
void pooling_ncsp(const kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, const void *po_rhs) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t batch = jpp.mb, channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padF = jpp.f_pad, padT = jpp.t_pad, padL = jpp.l_pad;
    const alg_kind_t alg = jpp.alg;
    const bool with_relu = jpp.with_relu;
    const float relu_alpha = jpp.relu_alpha;

    const size_t spatial_size = (size_t)inD * inH * inW;
    const size_t dst_spatial_size = (size_t)outD * outH * outW;

    // Binary post-ops are applied only on the channel-vectorized path, so when
    // a binary post-op is fused force every column through pool_column_vec_c by
    // emptying the OW-vectorized interior range.
    const dim_t int_lo = jpp.fuse_binary
            ? 0
            : std::max((padL + strideW - 1) / strideW, (dim_t)0);
    // See pooling_nspc: avoid truncating-division over-count when the kernel
    // is wider than the padded input (no full-window column then).
    const dim_t int_hi = jpp.fuse_binary
            ? 0
            : ((inW + padL >= kerW)
                              ? std::min(std::max((inW - kerW + padL) / strideW
                                                         + 1,
                                                 int_lo),
                                        outW)
                              : int_lo);

    // Interior columns: vectorize along OW.
    if (int_lo < int_hi) {
        parallel_nd(batch, channels, outD, outH,
                [&](dim_t mb, dim_t c, dim_t od, dim_t oh) {
            const data_t *src_ch
                    = src + ((size_t)mb * channels + c) * spatial_size;
            data_t *dst_oh = dst
                    + ((size_t)mb * channels + c) * dst_spatial_size
                    + od * outH * outW + oh * outW;

            const int id_start = std::max((int)(od * strideD - padF), 0);
            const int id_end
                    = std::min((int)(od * strideD - padF + kerD), (int)inD);
            const int ih_start = std::max((int)(oh * strideH - padT), 0);
            const int ih_end
                    = std::min((int)(oh * strideH - padT + kerH), (int)inH);

            // Whole (od, oh) row in front/top or back/bottom padding: every
            // interior column is an empty window. With no fused post-op, fill
            // them directly and skip the JIT call (big win on heavily-padded
            // shapes). With a fused chain, fall through so the kernel applies the
            // post-op to the init value (avg scale is 0 for an empty window, so
            // the result is post_op(0); max is post_op(lowest)) — matching the
            // call_one paths and x86/ARM, which always run the kernel.
            if ((id_start >= id_end || ih_start >= ih_end) && !jpp.fuse_eltwise
                    && !jpp.fuse_binary) {
                const data_t ev = empty_window_value<data_t>(jpp);
                for (dim_t ow = int_lo; ow < int_hi; ow++)
                    dst_oh[ow] = ev;
                return;
            }

            auto iv = compute_init_scale(alg, id_start, id_end, ih_start,
                    ih_end, 0, (int)kerW, kerD, kerH, kerW);

            jit_uni_pooling_args_t p;
            p.src = src_ch + (int_lo * strideW - padL);
            p.dst = dst_oh + int_lo;
            // 'channels' is repurposed as the interior OW count: the kernel's
            // vector loop walks OW with src_vec_byte_stride between positions.
            p.channels = int_hi - int_lo;
            p.id_start = id_start;
            p.ih_start = ih_start;
            p.iw_start = 0;
            p.id_end = id_end;
            p.ih_end = ih_end;
            p.iw_end = kerW;
            p.inW_stride = inW;
            p.inD_stride = inH * inW;
            p.w_spatial_byte_stride = (dim_t)sizeof(data_t);
            p.init_val = iv.init_val;
            p.scale_val = iv.scale_val;
            p.relu_alpha = relu_alpha;
            p.with_relu = with_relu;
            p.src_vec_byte_stride = strideW * (dim_t)sizeof(data_t);
            p.dst_vec_byte_stride = (dim_t)sizeof(data_t);
            kernel(&p);
        });
    }

    // Boundary columns: [0, lo) on the left, [hi, outW) on the right.
    const dim_t lo = std::min(int_lo, outW);
    const dim_t hi = (int_lo < int_hi) ? int_hi : lo;
    const dim_t n_bnd = lo + (outW - hi);
    if (n_bnd > 0) {
        parallel_nd(batch, outD, outH, n_bnd,
                [&](dim_t mb, dim_t od, dim_t oh, dim_t k) {
            const dim_t ow = (k < lo) ? k : hi + (k - lo);
            pool_column_vec_c<isa, d_type>(
                    kernel, jpp, src, dst, mb, od, oh, ow, po_rhs);
        });
    }
}

// ncsp, OW == 1: every column is handled by the vectorize-along-C path.
template <cpu_isa_t isa, data_type_t d_type>
void pooling_ncsp_ow1(const kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, const void *po_rhs) {
    parallel_nd(jpp.mb, jpp.od, jpp.oh, [&](dim_t mb, dim_t od, dim_t oh) {
        pool_column_vec_c<isa, d_type>(
                kernel, jpp, src, dst, mb, od, oh, 0, po_rhs);
    });
}

// nspc without a shape-baked interior kernel: one row-unrolled heavy call per
// interior OW run + one call per boundary column (the pre-bake path). Used for
// data types that have no baked interior kernel (f16).
template <cpu_isa_t isa, data_type_t d_type>
void pooling_nspc_generic(const kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, const void *po_rhs) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t batch = jpp.mb, channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padFront = jpp.f_pad, padTop = jpp.t_pad, padLeft = jpp.l_pad;
    const alg_kind_t alg = jpp.alg;
    const bool with_relu = jpp.with_relu;
    const float relu_alpha = jpp.relu_alpha;

    const dim_t inW_stride = inW * channels;
    const dim_t inD_stride = inH * inW * channels;

    const dim_t int_lo = std::max((padLeft + strideW - 1) / strideW, (dim_t)0);
    // Guard against truncating division: when inW + padLeft < kerW (kernel
    // wider than the padded input) the numerator is negative and integer
    // division would round toward zero, over-counting int_hi and marking a
    // partial window as full. In that case no full-window column exists.
    const dim_t int_hi = (inW + padLeft >= kerW)
            ? std::min(std::max((inW - kerW + padLeft) / strideW + 1, int_lo),
                      outW)
            : int_lo;

    parallel_nd(batch, outD, outH, [&](dim_t mb, dim_t od, dim_t oh) {
        const size_t mb_offset = (size_t)mb * (size_t)inD * inD_stride;
        const data_t *src_mb_base = &src[mb_offset];

        auto call_one = [&](dim_t ow) {
            auto p = make_pooling_params(alg, src_mb_base, dst, mb, channels,
                    od, oh, ow, outD, outH, outW, inD, inH, inW, kerD, kerH,
                    kerW, strideD, strideH, strideW, padFront, padTop, padLeft,
                    inW_stride, inD_stride, with_relu, relu_alpha);
            if ((p.id_start >= p.id_end || p.ih_start >= p.ih_end
                        || p.iw_start >= p.iw_end)
                    && !jpp.fuse_eltwise && !jpp.fuse_binary) {
                // Empty window, no fused post-op: fill channels, skip the kernel.
                const data_t ev = empty_window_value<data_t>(jpp);
                data_t *d = static_cast<data_t *>(p.dst);
                for (dim_t c = 0; c < channels; c++)
                    d[c] = ev;
                return;
            }
            p.post_op_rhs = binary_rhs_for(jpp, po_rhs, dst, p.dst);
            kernel(&p);
        };

        const dim_t lo = std::min(int_lo, outW);
        for (dim_t ow = 0; ow < lo; ++ow)
            call_one(ow);

        if (int_lo < int_hi) {
            auto p = make_pooling_params(alg, src_mb_base, dst, mb, channels,
                    od, oh, int_lo, outD, outH, outW, inD, inH, inW, kerD, kerH,
                    kerW, strideD, strideH, strideW, padFront, padTop, padLeft,
                    inW_stride, inD_stride, with_relu, relu_alpha);
            p.n_pos = int_hi - int_lo;
            p.pos_src_byte_stride
                    = (dim_t)strideW * channels * (dim_t)sizeof(data_t);
            p.pos_dst_byte_stride = (dim_t)channels * (dim_t)sizeof(data_t);
            kernel(&p);
        }

        const dim_t hi = (int_lo < int_hi) ? int_hi : lo;
        for (dim_t ow = hi; ow < outW; ++ow)
            call_one(ow);
    });
}

// Compile-time selection between the f32 shape-baked interior path and the
// generic nspc path (f16). C++11 has no `if constexpr`, so this uses a struct
// partial specialization to keep the f32-only baked call from being
// instantiated for f16 (where the pointer types would not match).
template <cpu_isa_t isa, data_type_t d_type>
struct nspc_exec {
    static void run(const kernel_t<isa, d_type> &k,
            const jit_uni_pool_interior_kernel_t<isa, d_type> *ik,
            const jit_pool_conf_t &jpp,
            const typename prec_traits_t<d_type>::type *src,
            typename prec_traits_t<d_type>::type *dst, const void *po_rhs) {
        UNUSED(ik);
        pooling_nspc_generic<isa, d_type>(k, jpp, src, dst, po_rhs);
    }
};

template <cpu_isa_t isa>
struct nspc_exec<isa, data_type::f32> {
    static void run(const kernel_t<isa, data_type::f32> &k,
            const jit_uni_pool_interior_kernel_t<isa, data_type::f32> *ik,
            const jit_pool_conf_t &jpp, const float *src, float *dst,
            const void *po_rhs) {
        pooling_nspc<isa, data_type::f32>(k, *ik, jpp, src, dst, po_rhs);
    }
};

} // namespace

template <cpu_isa_t isa>
jit_uni_pooling_fwd_t<isa>::jit_uni_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_pooling_fwd_t<isa>::~jit_uni_pooling_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_pooling_fwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);
    CHECK(safe_ptr_assign(
            kernel_, new jit_uni_pool_kernel_t<isa, d_type>(pd()->jpp_)));
    // The shape-baked interior kernel is f32-only; f16 uses the generic path.
    if (d_type == data_type::f32
            && pd()->jpp_.tag_kind == jit_pool_tag_kind_t::nspc)
        CHECK(safe_ptr_assign(interior_kernel_,
                new jit_uni_pool_interior_kernel_t<isa, d_type>(pd()->jpp_)));
    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_pooling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src_raw = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst_raw = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const size_t dt_size = types::data_type_size(src_d.data_type());

    src_raw = (const uint8_t *)src_raw + src_d.off_l(0) * dt_size;
    dst_raw = (uint8_t *)dst_raw + dst_d.off_l(0) * dt_size;

    const data_t *src = static_cast<const data_t *>(src_raw);
    data_t *dst = static_cast<data_t *>(dst_raw);

    const auto &jpp = pd()->jpp_;

    // Binary post-op src1 base (the lone binary in the fused chain when
    // fuse_binary); the kernels position/advance it per output position and
    // channel chunk. The binary may sit anywhere in the chain, so use its index.
    const void *po_rhs = nullptr;
    if (jpp.fuse_binary) {
        int bin_idx = 0;
        for (int i = 0; i < jpp.post_ops.len(); i++)
            if (jpp.post_ops.entry_[i].is_binary()) bin_idx = i;
        po_rhs = ctx.host_ptr(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_idx) | DNNL_ARG_SRC_1);
        // dst was shifted to its logical element 0 (off_l(0) above), and
        // binary_rhs_for()/the kernel address the rhs relative to that logical
        // layout (full-dst uses rhs + (p_dst - dst)). Shift the rhs base by
        // src1's own offset0 so the two are consistent — nonzero only for
        // submemory src1 descriptors, but otherwise it would read the wrong
        // address.
        const memory_desc_wrapper s1_d(
                jpp.post_ops.entry_[bin_idx].binary.src1_desc);
        po_rhs = (const uint8_t *)po_rhs + s1_d.off_l(0) * dt_size;
    }

    if (jpp.tag_kind == jit_pool_tag_kind_t::nspc) {
        // f32 uses the shape-baked interior kernel; f16 uses the generic path.
        nspc_exec<isa, d_type>::run(
                *kernel_, interior_kernel_.get(), jpp, src, dst, po_rhs);
    } else { // ncsp
        if (jpp.ow == 1)
            pooling_ncsp_ow1<isa, d_type>(*kernel_, jpp, src, dst, po_rhs);
        else
            pooling_ncsp<isa, d_type>(*kernel_, jpp, src, dst, po_rhs);
    }

    // All accepted post-ops are fused in the kernel; nothing to apply here.
    return status::success;
}

template struct jit_uni_pooling_fwd_t<v>;
template struct jit_uni_pooling_fwd_t<zvfh>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
