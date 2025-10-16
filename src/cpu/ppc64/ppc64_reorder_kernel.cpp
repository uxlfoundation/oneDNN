#include "cpu/ppc64/ppc64_reorder_kernel.hpp"
#include <altivec.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dnnl {
namespace impl {
namespace cpu {
namespace ppc64 {
namespace tr {

inline int8_t qz_s8(float d) {
    return static_cast<int8_t>(std::max(-128.f, std::min(127.f, std::round(d))));
}

inline uint8_t qz_u8(float d) {
    return static_cast<uint8_t>(std::max(0.f, std::min(255.f, std::round(d))));
}

reorder_s8_s8_kernel_t::reorder_s8_s8_kernel_t(const desc_t &desc) : kernel_t(desc) {}

bool reorder_s8_s8_kernel_t::applicable(const prb_t &prb) {
    return prb.itype == data_type::s8 && prb.otype == data_type::s8
        && prb.ioff == 0 && prb.ooff == 0
        && (prb.beta == 0.f || prb.beta == 1.f);
}

layout_kind_t reorder_s8_s8_kernel_t::detect_layout(const node_t &n0, const node_t &n1) const {
    if (n0.is == n1.n && n0.os == n1.n && n1.is == 1 && n1.os == 1)
        return layout_kind_t::row_major;
    if (n0.is == n1.n && n1.os == n0.n && n1.is == 1 && n0.os == 1)
        return layout_kind_t::col_major;
    if (n0.n == n1.n && n1.is == 1 && n0.os == 1)
        return layout_kind_t::blocked_transpose;
    return layout_kind_t::fallback_scalar;
}

void reorder_s8_s8_kernel_t::operator()(const call_param_t *p) const {
    const int8_t *src = reinterpret_cast<const int8_t *>(p->in);
    int8_t *dst = reinterpret_cast<int8_t *>(p->out);
    float src_scale = p->src_scales ? p->src_scales[0] : 1.f;
    float dst_scale = p->dst_scales ? p->dst_scales[0] : 1.f;
    int src_zp = p->src_zp;
    int dst_zp = p->dst_zp;
    float beta = prb_.beta;
    int32_t* compensation = p->compensation_scratch;
    copy_nd(src, dst, prb_.nodes, prb_.ndims, src_scale, dst_scale, src_zp, dst_zp, beta, compensation);
}

void reorder_s8_s8_kernel_t::operator()(const tail_call_param_t *p) const {
    const int8_t *src = reinterpret_cast<const int8_t *>(p->base_params.in);
    int8_t *dst = reinterpret_cast<int8_t *>(p->base_params.out);
    float src_scale = p->base_params.src_scales ? p->base_params.src_scales[0] : 1.f;
    float dst_scale = p->base_params.dst_scales ? p->base_params.dst_scales[0] : 1.f;
    int src_zp = p->base_params.src_zp;
    int dst_zp = p->base_params.dst_zp;
    float beta = prb_.beta;
    int32_t* compensation = p->base_params.compensation_scratch;
    std::vector<node_t> nodes(prb_.nodes, prb_.nodes + prb_.ndims);
    for (int d = 0; d < prb_.ndims; ++d)
        if (nodes[d].tail_size > 0) nodes[d].n = nodes[d].tail_size;
    copy_nd(src, dst, nodes.data(), prb_.ndims, src_scale, dst_scale, src_zp, dst_zp, beta, compensation);
}

void reorder_s8_s8_kernel_t::copy_nd(const int8_t* src, int8_t* dst, const node_t* nodes, int ndims,
                                     float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int32_t* compensation) const {
    if (ndims == 1) {
        copy_1d(src, dst, nodes[0], src_scale, dst_scale, src_zp, dst_zp, beta, compensation);
    } else if (ndims == 2) {
        copy_2d(src, dst, nodes[0], nodes[1], src_scale, dst_scale, src_zp, dst_zp, beta, compensation);
    } else {
        // ND: loop over outer dims, call 2D block kernel for inner tile
        size_t outer_elems = 1;
        for (int d = 2; d < ndims; ++d) outer_elems *= nodes[d].n;
        std::vector<size_t> idx(ndims, 0);
        for (size_t off = 0; off < outer_elems; ++off) {
            ptrdiff_t src_offset = 0, dst_offset = 0;
            for (int d = 2; d < ndims; ++d) {
                src_offset += idx[d] * nodes[d].is;
                dst_offset += idx[d] * nodes[d].os;
            }
            int32_t* comp_ptr = compensation ? &compensation[off] : nullptr;
            copy_2d(&src[src_offset], &dst[dst_offset], nodes[0], nodes[1], src_scale, dst_scale, src_zp, dst_zp, beta, comp_ptr);
            for (int d = ndims - 1; d >= 2; --d) {
                if (++idx[d] < nodes[d].n) break;
                idx[d] = 0;
            }
        }
    }
}

void reorder_s8_s8_kernel_t::copy_1d(const int8_t* src, int8_t* dst, const node_t& n0,
                                      float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int32_t* compensation) const {
     for (int i = 0; i < n0.n; ++i) {
         float val = src_scale * (src[i * n0.is] - src_zp);
         if (beta) val += beta * dst[i * n0.os];
         val = val * dst_scale + dst_zp;
         dst[i * n0.os] = qz_s8(val);
	 }
}

void reorder_s8_s8_kernel_t::copy_2d(const int8_t* src, int8_t* dst, const node_t &n0, const node_t &n1,
                                     float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int32_t* compensation) const {
    layout_kind_t layout = detect_layout(n0, n1);
    int M = n0.n, N = n1.n, src_stride = n0.is, dst_stride = n1.os;
    switch (layout) {
        case layout_kind_t::row_major:
            reorder_row_major_kernel(src, dst, M, N, src_stride, dst_stride, src_scale, dst_scale, src_zp, dst_zp, beta, compensation);
            break;
        case layout_kind_t::col_major:
        case layout_kind_t::blocked_transpose:
            reorder_col_major_kernel(src, dst, M, N, src_stride, dst_stride, src_scale, dst_scale, src_zp, dst_zp, beta, compensation);
            break;
        case layout_kind_t::fallback_scalar:
        default:
            scalar_tail_kernel(src, dst, M, N, src_stride, dst_stride, src_scale, dst_scale, src_zp, dst_zp, beta, 0, compensation);
            break;
    }
}

// --- Block kernel implementations ---
// Vectorized 16x16, 8x8, 4x4 block kernels for s8->s8
// s8->s8 block kernels: correct int8->float conversion for all 16 elements
void reorder_s8_s8_kernel_t::block_kernel_16x16_row_major(const int8_t* src, int8_t* dst, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation) {
    for (int i = 0; i < 16; ++i) {
        __vector signed char vsrc = vec_vsx_ld(0, &src[i * src_stride]);
        __vector signed short vsrc_lo = vec_unpackh(vsrc);
        __vector signed short vsrc_hi = vec_unpackl(vsrc);
        __vector signed int vsrc_ll = vec_unpackh(vsrc_lo);
        __vector signed int vsrc_lh = vec_unpackl(vsrc_lo);
        __vector signed int vsrc_hl = vec_unpackh(vsrc_hi);
        __vector signed int vsrc_hh = vec_unpackl(vsrc_hi);
        __vector float vf_ll = vec_ctf(vsrc_ll, 0);
        __vector float vf_lh = vec_ctf(vsrc_lh, 0);
        __vector float vf_hl = vec_ctf(vsrc_hl, 0);
        __vector float vf_hh = vec_ctf(vsrc_hh, 0);
        // Apply quantization math
        vf_ll = vec_sub(vf_ll, vec_splats((float)src_zp));
        vf_lh = vec_sub(vf_lh, vec_splats((float)src_zp));
        vf_hl = vec_sub(vf_hl, vec_splats((float)src_zp));
        vf_hh = vec_sub(vf_hh, vec_splats((float)src_zp));
        vf_ll = vec_mul(vf_ll, vec_splats(src_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(src_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(src_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(src_scale));
        if (beta) {
            __vector signed char vdst = vec_vsx_ld(0, &dst[i * dst_stride]);
            __vector signed short vdst_lo = vec_unpackh(vdst);
            __vector signed short vdst_hi = vec_unpackl(vdst);
            __vector signed int vdst_ll = vec_unpackh(vdst_lo);
            __vector signed int vdst_lh = vec_unpackl(vdst_lo);
            __vector signed int vdst_hl = vec_unpackh(vdst_hi);
            __vector signed int vdst_hh = vec_unpackl(vdst_hi);
            __vector float vdf_ll = vec_ctf(vdst_ll, 0);
            __vector float vdf_lh = vec_ctf(vdst_lh, 0);
            __vector float vdf_hl = vec_ctf(vdst_hl, 0);
            __vector float vdf_hh = vec_ctf(vdst_hh, 0);
            vf_ll = vec_add(vf_ll, vec_mul(vdf_ll, vec_splats(beta)));
            vf_lh = vec_add(vf_lh, vec_mul(vdf_lh, vec_splats(beta)));
            vf_hl = vec_add(vf_hl, vec_mul(vdf_hl, vec_splats(beta)));
            vf_hh = vec_add(vf_hh, vec_mul(vdf_hh, vec_splats(beta)));
        }
        vf_ll = vec_mul(vf_ll, vec_splats(dst_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(dst_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(dst_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(dst_scale));
        vf_ll = vec_add(vf_ll, vec_splats((float)dst_zp));
        vf_lh = vec_add(vf_lh, vec_splats((float)dst_zp));
        vf_hl = vec_add(vf_hl, vec_splats((float)dst_zp));
        vf_hh = vec_add(vf_hh, vec_splats((float)dst_zp));
        vf_ll = vec_max(vf_ll, vec_splats(-128.f));
        vf_ll = vec_min(vf_ll, vec_splats(127.f));
        vf_lh = vec_max(vf_lh, vec_splats(-128.f));
        vf_lh = vec_min(vf_lh, vec_splats(127.f));
        vf_hl = vec_max(vf_hl, vec_splats(-128.f));
        vf_hl = vec_min(vf_hl, vec_splats(127.f));
        vf_hh = vec_max(vf_hh, vec_splats(-128.f));
        vf_hh = vec_min(vf_hh, vec_splats(127.f));
        __vector signed int vi_ll = vec_cts(vf_ll, 0);
        __vector signed int vi_lh = vec_cts(vf_lh, 0);
        __vector signed int vi_hl = vec_cts(vf_hl, 0);
        __vector signed int vi_hh = vec_cts(vf_hh, 0);
        __vector signed short vs_lo = vec_packs(vi_ll, vi_lh);
        __vector signed short vs_hi = vec_packs(vi_hl, vi_hh);
        __vector signed char vout = vec_packs(vs_lo, vs_hi);
        vec_vsx_st(vout, 0, &dst[i * dst_stride]);
        if (compensation) {
            int32_t comp = 0;
            for (int k = 0; k < 16; ++k)
                comp += src[i * src_stride + k];
            compensation[comp_offset + i] = comp;
        }
    }
}
void reorder_s8_s8_kernel_t::block_kernel_8x8_row_major(const int8_t* src, int8_t* dst, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation) {
    for (int i = 0; i < 8; ++i) {
        __vector signed char vsrc = vec_vsx_ld(0, &src[i * src_stride]);
        __vector signed short vsrc_lo = vec_unpackh(vsrc);
        __vector signed short vsrc_hi = vec_unpackl(vsrc);
        __vector signed int vsrc_ll = vec_unpackh(vsrc_lo);
        __vector signed int vsrc_lh = vec_unpackl(vsrc_lo);
        __vector signed int vsrc_hl = vec_unpackh(vsrc_hi);
        __vector signed int vsrc_hh = vec_unpackl(vsrc_hi);
        __vector float vf_ll = vec_ctf(vsrc_ll, 0);
        __vector float vf_lh = vec_ctf(vsrc_lh, 0);
        __vector float vf_hl = vec_ctf(vsrc_hl, 0);
        __vector float vf_hh = vec_ctf(vsrc_hh, 0);
        // Apply quantization math
        vf_ll = vec_sub(vf_ll, vec_splats((float)src_zp));
        vf_lh = vec_sub(vf_lh, vec_splats((float)src_zp));
        vf_hl = vec_sub(vf_hl, vec_splats((float)src_zp));
        vf_hh = vec_sub(vf_hh, vec_splats((float)src_zp));
        vf_ll = vec_mul(vf_ll, vec_splats(src_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(src_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(src_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(src_scale));
        if (beta) {
            __vector signed char vdst = vec_vsx_ld(0, &dst[i * dst_stride]);
            __vector signed short vdst_lo = vec_unpackh(vdst);
            __vector signed short vdst_hi = vec_unpackl(vdst);
            __vector signed int vdst_ll = vec_unpackh(vdst_lo);
            __vector signed int vdst_lh = vec_unpackl(vdst_lo);
            __vector signed int vdst_hl = vec_unpackh(vdst_hi);
            __vector signed int vdst_hh = vec_unpackl(vdst_hi);
            __vector float vdf_ll = vec_ctf(vdst_ll, 0);
            __vector float vdf_lh = vec_ctf(vdst_lh, 0);
            __vector float vdf_hl = vec_ctf(vdst_hl, 0);
            __vector float vdf_hh = vec_ctf(vdst_hh, 0);
            vf_ll = vec_add(vf_ll, vec_mul(vdf_ll, vec_splats(beta)));
            vf_lh = vec_add(vf_lh, vec_mul(vdf_lh, vec_splats(beta)));
            vf_hl = vec_add(vf_hl, vec_mul(vdf_hl, vec_splats(beta)));
            vf_hh = vec_add(vf_hh, vec_mul(vdf_hh, vec_splats(beta)));
        }
        vf_ll = vec_mul(vf_ll, vec_splats(dst_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(dst_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(dst_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(dst_scale));
        vf_ll = vec_add(vf_ll, vec_splats((float)dst_zp));
        vf_lh = vec_add(vf_lh, vec_splats((float)dst_zp));
        vf_hl = vec_add(vf_hl, vec_splats((float)dst_zp));
        vf_hh = vec_add(vf_hh, vec_splats((float)dst_zp));
        vf_ll = vec_max(vf_ll, vec_splats(-128.f));
        vf_ll = vec_min(vf_ll, vec_splats(127.f));
        vf_lh = vec_max(vf_lh, vec_splats(-128.f));
        vf_lh = vec_min(vf_lh, vec_splats(127.f));
        vf_hl = vec_max(vf_hl, vec_splats(-128.f));
        vf_hl = vec_min(vf_hl, vec_splats(127.f));
        vf_hh = vec_max(vf_hh, vec_splats(-128.f));
        vf_hh = vec_min(vf_hh, vec_splats(127.f));
        __vector signed int vi_ll = vec_cts(vf_ll, 0);
        __vector signed int vi_lh = vec_cts(vf_lh, 0);
        __vector signed int vi_hl = vec_cts(vf_hl, 0);
        __vector signed int vi_hh = vec_cts(vf_hh, 0);
        __vector signed short vs_lo = vec_packs(vi_ll, vi_lh);
        __vector signed short vs_hi = vec_packs(vi_hl, vi_hh);
        __vector signed char vout = vec_packs(vs_lo, vs_hi);
        vec_vsx_st(vout, 0, &dst[i * dst_stride]);
        if (compensation) {
            int32_t comp = 0;
            for (int k = 0; k < 8; ++k)
                comp += src[i * src_stride + k];
            compensation[comp_offset + i] = comp;
        }
    }
}
void reorder_s8_s8_kernel_t::block_kernel_4x4_row_major(const int8_t* src, int8_t* dst, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation) {
    for (int i = 0; i < 4; ++i) {
        __vector signed char vsrc = vec_vsx_ld(0, &src[i * src_stride]);
        __vector signed short vsrc_lo = vec_unpackh(vsrc);
        __vector signed short vsrc_hi = vec_unpackl(vsrc);
        __vector signed int vsrc_ll = vec_unpackh(vsrc_lo);
        __vector signed int vsrc_lh = vec_unpackl(vsrc_lo);
        __vector signed int vsrc_hl = vec_unpackh(vsrc_hi);
        __vector signed int vsrc_hh = vec_unpackl(vsrc_hi);
        __vector float vf_ll = vec_ctf(vsrc_ll, 0);
        __vector float vf_lh = vec_ctf(vsrc_lh, 0);
        __vector float vf_hl = vec_ctf(vsrc_hl, 0);
        __vector float vf_hh = vec_ctf(vsrc_hh, 0);
        // Apply quantization math
        vf_ll = vec_sub(vf_ll, vec_splats((float)src_zp));
        vf_lh = vec_sub(vf_lh, vec_splats((float)src_zp));
        vf_hl = vec_sub(vf_hl, vec_splats((float)src_zp));
        vf_hh = vec_sub(vf_hh, vec_splats((float)src_zp));
        vf_ll = vec_mul(vf_ll, vec_splats(src_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(src_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(src_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(src_scale));
        if (beta) {
            __vector signed char vdst = vec_vsx_ld(0, &dst[i * dst_stride]);
            __vector signed short vdst_lo = vec_unpackh(vdst);
            __vector signed short vdst_hi = vec_unpackl(vdst);
            __vector signed int vdst_ll = vec_unpackh(vdst_lo);
            __vector signed int vdst_lh = vec_unpackl(vdst_lo);
            __vector signed int vdst_hl = vec_unpackh(vdst_hi);
            __vector signed int vdst_hh = vec_unpackl(vdst_hi);
            __vector float vdf_ll = vec_ctf(vdst_ll, 0);
            __vector float vdf_lh = vec_ctf(vdst_lh, 0);
            __vector float vdf_hl = vec_ctf(vdst_hl, 0);
            __vector float vdf_hh = vec_ctf(vdst_hh, 0);
            vf_ll = vec_add(vf_ll, vec_mul(vdf_ll, vec_splats(beta)));
            vf_lh = vec_add(vf_lh, vec_mul(vdf_lh, vec_splats(beta)));
            vf_hl = vec_add(vf_hl, vec_mul(vdf_hl, vec_splats(beta)));
            vf_hh = vec_add(vf_hh, vec_mul(vdf_hh, vec_splats(beta)));
        }
        vf_ll = vec_mul(vf_ll, vec_splats(dst_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(dst_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(dst_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(dst_scale));
        vf_ll = vec_add(vf_ll, vec_splats((float)dst_zp));
        vf_lh = vec_add(vf_lh, vec_splats((float)dst_zp));
        vf_hl = vec_add(vf_hl, vec_splats((float)dst_zp));
        vf_hh = vec_add(vf_hh, vec_splats((float)dst_zp));
        vf_ll = vec_max(vf_ll, vec_splats(-128.f));
        vf_ll = vec_min(vf_ll, vec_splats(127.f));
        vf_lh = vec_max(vf_lh, vec_splats(-128.f));
        vf_lh = vec_min(vf_lh, vec_splats(127.f));
        vf_hl = vec_max(vf_hl, vec_splats(-128.f));
        vf_hl = vec_min(vf_hl, vec_splats(127.f));
        vf_hh = vec_max(vf_hh, vec_splats(-128.f));
        vf_hh = vec_min(vf_hh, vec_splats(127.f));
        __vector signed int vi_ll = vec_cts(vf_ll, 0);
        __vector signed int vi_lh = vec_cts(vf_lh, 0);
        __vector signed int vi_hl = vec_cts(vf_hl, 0);
        __vector signed int vi_hh = vec_cts(vf_hh, 0);
        __vector signed short vs_lo = vec_packs(vi_ll, vi_lh);
        __vector signed short vs_hi = vec_packs(vi_hl, vi_hh);
        __vector signed char vout = vec_packs(vs_lo, vs_hi);
        vec_vsx_st(vout, 0, &dst[i * dst_stride]);
        if (compensation) {
            int32_t comp = 0;
            for (int k = 0; k < 4; ++k)
                comp += src[i * src_stride + k];
            compensation[comp_offset + i] = comp;
        }
    }
}
void reorder_s8_s8_kernel_t::block_kernel_16x16_col_major(const int8_t* src, int8_t* dst, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation) {
    for (int j = 0; j < 16; ++j) {
        __vector signed char vsrc = vec_vsx_ld(0, &src[j]);
        __vector signed short vsrc_lo = vec_unpackh(vsrc);
        __vector signed short vsrc_hi = vec_unpackl(vsrc);
        __vector signed int vsrc_ll = vec_unpackh(vsrc_lo);
        __vector signed int vsrc_lh = vec_unpackl(vsrc_lo);
        __vector signed int vsrc_hl = vec_unpackh(vsrc_hi);
        __vector signed int vsrc_hh = vec_unpackl(vsrc_hi);
        __vector float vf_ll = vec_ctf(vsrc_ll, 0);
        __vector float vf_lh = vec_ctf(vsrc_lh, 0);
        __vector float vf_hl = vec_ctf(vsrc_hl, 0);
        __vector float vf_hh = vec_ctf(vsrc_hh, 0);
        // Apply quantization math
        vf_ll = vec_sub(vf_ll, vec_splats((float)src_zp));
        vf_lh = vec_sub(vf_lh, vec_splats((float)src_zp));
        vf_hl = vec_sub(vf_hl, vec_splats((float)src_zp));
        vf_hh = vec_sub(vf_hh, vec_splats((float)src_zp));
        vf_ll = vec_mul(vf_ll, vec_splats(src_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(src_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(src_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(src_scale));
        if (beta) {
            __vector signed char vdst = vec_vsx_ld(0, &dst[j * dst_stride]);
            __vector signed short vdst_lo = vec_unpackh(vdst);
            __vector signed short vdst_hi = vec_unpackl(vdst);
            __vector signed int vdst_ll = vec_unpackh(vdst_lo);
            __vector signed int vdst_lh = vec_unpackl(vdst_lo);
            __vector signed int vdst_hl = vec_unpackh(vdst_hi);
            __vector signed int vdst_hh = vec_unpackl(vdst_hi);
            __vector float vdf_ll = vec_ctf(vdst_ll, 0);
            __vector float vdf_lh = vec_ctf(vdst_lh, 0);
            __vector float vdf_hl = vec_ctf(vdst_hl, 0);
            __vector float vdf_hh = vec_ctf(vdst_hh, 0);
            vf_ll = vec_add(vf_ll, vec_mul(vdf_ll, vec_splats(beta)));
            vf_lh = vec_add(vf_lh, vec_mul(vdf_lh, vec_splats(beta)));
            vf_hl = vec_add(vf_hl, vec_mul(vdf_hl, vec_splats(beta)));
            vf_hh = vec_add(vf_hh, vec_mul(vdf_hh, vec_splats(beta)));
        }
        vf_ll = vec_mul(vf_ll, vec_splats(dst_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(dst_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(dst_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(dst_scale));
        vf_ll = vec_add(vf_ll, vec_splats((float)dst_zp));
        vf_lh = vec_add(vf_lh, vec_splats((float)dst_zp));
        vf_hl = vec_add(vf_hl, vec_splats((float)dst_zp));
        vf_hh = vec_add(vf_hh, vec_splats((float)dst_zp));
        vf_ll = vec_max(vf_ll, vec_splats(-128.f));
        vf_ll = vec_min(vf_ll, vec_splats(127.f));
        vf_lh = vec_max(vf_lh, vec_splats(-128.f));
        vf_lh = vec_min(vf_lh, vec_splats(127.f));
        vf_hl = vec_max(vf_hl, vec_splats(-128.f));
        vf_hl = vec_min(vf_hl, vec_splats(127.f));
        vf_hh = vec_max(vf_hh, vec_splats(-128.f));
        vf_hh = vec_min(vf_hh, vec_splats(127.f));
        __vector signed int vi_ll = vec_cts(vf_ll, 0);
        __vector signed int vi_lh = vec_cts(vf_lh, 0);
        __vector signed int vi_hl = vec_cts(vf_hl, 0);
        __vector signed int vi_hh = vec_cts(vf_hh, 0);
        __vector signed short vs_lo = vec_packs(vi_ll, vi_lh);
        __vector signed short vs_hi = vec_packs(vi_hl, vi_hh);
        __vector signed char vout = vec_packs(vs_lo, vs_hi);
        vec_vsx_st(vout, 0, &dst[j * dst_stride]);
        if (compensation) {
            int32_t comp = 0;
            for (int k = 0; k < 16; ++k)
                comp += src[k * src_stride + j];
            compensation[comp_offset + j] = comp;
        }
    }
}
void reorder_s8_s8_kernel_t::block_kernel_8x8_col_major(const int8_t* src, int8_t* dst, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation) {
    for (int j = 0; j < 8; ++j) {
        __vector signed char vsrc = vec_vsx_ld(0, &src[j]);
        __vector signed short vsrc_lo = vec_unpackh(vsrc);
        __vector signed short vsrc_hi = vec_unpackl(vsrc);
        __vector signed int vsrc_ll = vec_unpackh(vsrc_lo);
        __vector signed int vsrc_lh = vec_unpackl(vsrc_lo);
        __vector signed int vsrc_hl = vec_unpackh(vsrc_hi);
        __vector signed int vsrc_hh = vec_unpackl(vsrc_hi);
        __vector float vf_ll = vec_ctf(vsrc_ll, 0);
        __vector float vf_lh = vec_ctf(vsrc_lh, 0);
        __vector float vf_hl = vec_ctf(vsrc_hl, 0);
        __vector float vf_hh = vec_ctf(vsrc_hh, 0);
        // Apply quantization math
        vf_ll = vec_sub(vf_ll, vec_splats((float)src_zp));
        vf_lh = vec_sub(vf_lh, vec_splats((float)src_zp));
        vf_hl = vec_sub(vf_hl, vec_splats((float)src_zp));
        vf_hh = vec_sub(vf_hh, vec_splats((float)src_zp));
        vf_ll = vec_mul(vf_ll, vec_splats(src_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(src_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(src_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(src_scale));
        if (beta) {
            __vector signed char vdst = vec_vsx_ld(0, &dst[j * dst_stride]);
            __vector signed short vdst_lo = vec_unpackh(vdst);
            __vector signed short vdst_hi = vec_unpackl(vdst);
            __vector signed int vdst_ll = vec_unpackh(vdst_lo);
            __vector signed int vdst_lh = vec_unpackl(vdst_lo);
            __vector signed int vdst_hl = vec_unpackh(vdst_hi);
            __vector signed int vdst_hh = vec_unpackl(vdst_hi);
            __vector float vdf_ll = vec_ctf(vdst_ll, 0);
            __vector float vdf_lh = vec_ctf(vdst_lh, 0);
            __vector float vdf_hl = vec_ctf(vdst_hl, 0);
            __vector float vdf_hh = vec_ctf(vdst_hh, 0);
            vf_ll = vec_add(vf_ll, vec_mul(vdf_ll, vec_splats(beta)));
            vf_lh = vec_add(vf_lh, vec_mul(vdf_lh, vec_splats(beta)));
            vf_hl = vec_add(vf_hl, vec_mul(vdf_hl, vec_splats(beta)));
            vf_hh = vec_add(vf_hh, vec_mul(vdf_hh, vec_splats(beta)));
        }
        vf_ll = vec_mul(vf_ll, vec_splats(dst_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(dst_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(dst_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(dst_scale));
        vf_ll = vec_add(vf_ll, vec_splats((float)dst_zp));
        vf_lh = vec_add(vf_lh, vec_splats((float)dst_zp));
        vf_hl = vec_add(vf_hl, vec_splats((float)dst_zp));
        vf_hh = vec_add(vf_hh, vec_splats((float)dst_zp));
        vf_ll = vec_max(vf_ll, vec_splats(-128.f));
        vf_ll = vec_min(vf_ll, vec_splats(127.f));
        vf_lh = vec_max(vf_lh, vec_splats(-128.f));
        vf_lh = vec_min(vf_lh, vec_splats(127.f));
        vf_hl = vec_max(vf_hl, vec_splats(-128.f));
        vf_hl = vec_min(vf_hl, vec_splats(127.f));
        vf_hh = vec_max(vf_hh, vec_splats(-128.f));
        vf_hh = vec_min(vf_hh, vec_splats(127.f));
        __vector signed int vi_ll = vec_cts(vf_ll, 0);
        __vector signed int vi_lh = vec_cts(vf_lh, 0);
        __vector signed int vi_hl = vec_cts(vf_hl, 0);
        __vector signed int vi_hh = vec_cts(vf_hh, 0);
        __vector signed short vs_lo = vec_packs(vi_ll, vi_lh);
        __vector signed short vs_hi = vec_packs(vi_hl, vi_hh);
        __vector signed char vout = vec_packs(vs_lo, vs_hi);
        vec_vsx_st(vout, 0, &dst[j * dst_stride]);
        if (compensation) {
            int32_t comp = 0;
            for (int k = 0; k < 8; ++k)
                comp += src[k * src_stride + j];
            compensation[comp_offset + j] = comp;
        }
    }
}
void reorder_s8_s8_kernel_t::block_kernel_4x4_col_major(const int8_t* src, int8_t* dst, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation) {
    for (int j = 0; j < 4; ++j) {
        __vector signed char vsrc = vec_vsx_ld(0, &src[j]);
        __vector signed short vsrc_lo = vec_unpackh(vsrc);
        __vector signed short vsrc_hi = vec_unpackl(vsrc);
        __vector signed int vsrc_ll = vec_unpackh(vsrc_lo);
        __vector signed int vsrc_lh = vec_unpackl(vsrc_lo);
        __vector signed int vsrc_hl = vec_unpackh(vsrc_hi);
        __vector signed int vsrc_hh = vec_unpackl(vsrc_hi);
        __vector float vf_ll = vec_ctf(vsrc_ll, 0);
        __vector float vf_lh = vec_ctf(vsrc_lh, 0);
        __vector float vf_hl = vec_ctf(vsrc_hl, 0);
        __vector float vf_hh = vec_ctf(vsrc_hh, 0);
        // Apply quantization math
        vf_ll = vec_sub(vf_ll, vec_splats((float)src_zp));
        vf_lh = vec_sub(vf_lh, vec_splats((float)src_zp));
        vf_hl = vec_sub(vf_hl, vec_splats((float)src_zp));
        vf_hh = vec_sub(vf_hh, vec_splats((float)src_zp));
        vf_ll = vec_mul(vf_ll, vec_splats(src_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(src_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(src_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(src_scale));
        if (beta) {
            __vector signed char vdst = vec_vsx_ld(0, &dst[j * dst_stride]);
            __vector signed short vdst_lo = vec_unpackh(vdst);
            __vector signed short vdst_hi = vec_unpackl(vdst);
            __vector signed int vdst_ll = vec_unpackh(vdst_lo);
            __vector signed int vdst_lh = vec_unpackl(vdst_lo);
            __vector signed int vdst_hl = vec_unpackh(vdst_hi);
            __vector signed int vdst_hh = vec_unpackl(vdst_hi);
            __vector float vdf_ll = vec_ctf(vdst_ll, 0);
            __vector float vdf_lh = vec_ctf(vdst_lh, 0);
            __vector float vdf_hl = vec_ctf(vdst_hl, 0);
            __vector float vdf_hh = vec_ctf(vdst_hh, 0);
            vf_ll = vec_add(vf_ll, vec_mul(vdf_ll, vec_splats(beta)));
            vf_lh = vec_add(vf_lh, vec_mul(vdf_lh, vec_splats(beta)));
            vf_hl = vec_add(vf_hl, vec_mul(vdf_hl, vec_splats(beta)));
            vf_hh = vec_add(vf_hh, vec_mul(vdf_hh, vec_splats(beta)));
        }
        vf_ll = vec_mul(vf_ll, vec_splats(dst_scale));
        vf_lh = vec_mul(vf_lh, vec_splats(dst_scale));
        vf_hl = vec_mul(vf_hl, vec_splats(dst_scale));
        vf_hh = vec_mul(vf_hh, vec_splats(dst_scale));
        vf_ll = vec_add(vf_ll, vec_splats((float)dst_zp));
        vf_lh = vec_add(vf_lh, vec_splats((float)dst_zp));
        vf_hl = vec_add(vf_hl, vec_splats((float)dst_zp));
        vf_hh = vec_add(vf_hh, vec_splats((float)dst_zp));
        vf_ll = vec_max(vf_ll, vec_splats(-128.f));
        vf_ll = vec_min(vf_ll, vec_splats(127.f));
        vf_lh = vec_max(vf_lh, vec_splats(-128.f));
        vf_lh = vec_min(vf_lh, vec_splats(127.f));
        vf_hl = vec_max(vf_hl, vec_splats(-128.f));
        vf_hl = vec_min(vf_hl, vec_splats(127.f));
        vf_hh = vec_max(vf_hh, vec_splats(-128.f));
        vf_hh = vec_min(vf_hh, vec_splats(127.f));
        __vector signed int vi_ll = vec_cts(vf_ll, 0);
        __vector signed int vi_lh = vec_cts(vf_lh, 0);
        __vector signed int vi_hl = vec_cts(vf_hl, 0);
        __vector signed int vi_hh = vec_cts(vf_hh, 0);
        __vector signed short vs_lo = vec_packs(vi_ll, vi_lh);
        __vector signed short vs_hi = vec_packs(vi_hl, vi_hh);
        __vector signed char vout = vec_packs(vs_lo, vs_hi);
        vec_vsx_st(vout, 0, &dst[j * dst_stride]);
        if (compensation) {
            int32_t comp = 0;
            for (int k = 0; k < 4; ++k)
                comp += src[k * src_stride + j];
            compensation[comp_offset + j] = comp;
        }
    }
}

void reorder_s8_s8_kernel_t::scalar_tail_kernel(const int8_t* src, int8_t* dst, int M, int N, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation) {
    for (int i = 0; i < M; ++i) {
        int32_t comp = 0;
        for (int j = 0; j < N; ++j) {
            float val = src_scale * (src[i * src_stride + j] - src_zp);
            if (beta) val += beta * dst[i * dst_stride + j];
            val = val * dst_scale + dst_zp;
            dst[i * dst_stride + j] = qz_s8(val);
            if (compensation) comp += static_cast<int32_t>(src[i * src_stride + j]);
        }
        if (compensation) compensation[comp_offset + i] = comp;
    }
}


void reorder_s8_s8_kernel_t::scalar_tail_kernel_col_major(
    const int8_t* src, int8_t* dst, int M, int N, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation)
{
    for (int i = 0; i < M; ++i) {
        int32_t comp = 0;
        for (int j = 0; j < N; ++j) {
            float val = src_scale * (src[i * src_stride + j] - src_zp);
            if (beta) val += beta * dst[j * dst_stride + i];
            val = val * dst_scale + dst_zp;
            dst[j * dst_stride + i] = qz_s8(val);
            if (compensation) comp += static_cast<int32_t>(src[i * src_stride + j]);
        }
        if (compensation) compensation[comp_offset + i] = comp;
    }
}
void reorder_s8_s8_kernel_t::scalar_tail_kernel_row_major(
    const int8_t* src, int8_t* dst, int M, int N, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int comp_offset, int32_t* compensation)
{
    for (int i = 0; i < M; ++i) {
        int32_t comp = 0;
        for (int j = 0; j < N; ++j) {
            float val = src_scale * (src[i * src_stride + j] - src_zp);
            if (beta) val += beta * dst[i * dst_stride + j];
            val = val * dst_scale + dst_zp;
            dst[i * dst_stride + j] = qz_s8(val);
            if (compensation) comp += static_cast<int32_t>(src[i * src_stride + j]);
        }
        if (compensation) compensation[comp_offset + i] = comp;
    }
}

// --- Tiling helpers ---
void reorder_s8_s8_kernel_t::reorder_row_major_kernel(const int8_t* src, int8_t* dst, int M, int N, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int32_t* compensation) const {
    int i = 0;
    while (i + 16 <= M) {
        int j = 0;
        while (j + 16 <= N) {
            block_kernel_16x16_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], src_stride, dst_stride,
                                         src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
            j += 16;
        }
        while (j + 8 <= N) {
            block_kernel_8x8_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
            j += 8;
        }
        while (j + 4 <= N) {
            block_kernel_4x4_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
            j += 4;
        }
        if (j < N)
            scalar_tail_kernel_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], 16, N - j, src_stride, dst_stride,
                               src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
        i += 16;
    }
    while (i + 8 <= M) {
        int j = 0;
        while (j + 8 <= N) {
            block_kernel_8x8_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
            j += 8;
        }
        while (j + 4 <= N) {
            block_kernel_4x4_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
            j += 4;
        }
        if (j < N)
            scalar_tail_kernel_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], 8, N - j, src_stride, dst_stride,
                               src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
        i += 8;
    }
    while (i + 4 <= M) {
        int j = 0;
        while (j + 4 <= N) {
            block_kernel_4x4_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
            j += 4;
        }
        if (j < N)
            scalar_tail_kernel_row_major(&src[i * src_stride + j], &dst[i * dst_stride + j], 4, N - j, src_stride, dst_stride,
                               src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
        i += 4;
    }
    if (i < M)
        scalar_tail_kernel_row_major(&src[i * src_stride], &dst[i * dst_stride], M - i, N, src_stride, dst_stride,
                           src_scale, dst_scale, src_zp, dst_zp, beta, i, compensation);
}

void reorder_s8_s8_kernel_t::reorder_col_major_kernel(const int8_t* src, int8_t* dst, int M, int N, int src_stride, int dst_stride,
    float src_scale, float dst_scale, int src_zp, int dst_zp, float beta, int32_t* compensation) const {
    int i = 0;
    while (i + 16 <= M) {
        int j = 0;
        while (j + 16 <= N) {
            block_kernel_16x16_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], src_stride, dst_stride,
                                         src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
            j += 16;
        }
        while (j + 8 <= N) {
            block_kernel_8x8_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
            j += 8;
        }
        while (j + 4 <= N) {
            block_kernel_4x4_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
            j += 4;
        }
        if (j < N)
            scalar_tail_kernel_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], 16, N - j, src_stride, dst_stride,
                               src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
        i += 16;
    }
    while (i + 8 <= M) {
        int j = 0;
        while (j + 8 <= N) {
            block_kernel_8x8_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
            j += 8;
        }
        while (j + 4 <= N) {
            block_kernel_4x4_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
            j += 4;
        }
        if (j < N)
            scalar_tail_kernel_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], 8, N - j, src_stride, dst_stride,
                               src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
        i += 8;
    }
    while (i + 4 <= M) {
        int j = 0;
        while (j + 4 <= N) {
            block_kernel_4x4_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], src_stride, dst_stride,
                                       src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
            j += 4;
        }
        if (j < N)
            scalar_tail_kernel_col_major(&src[i * src_stride + j], &dst[j * dst_stride + i], 4, N - j, src_stride, dst_stride,
                               src_scale, dst_scale, src_zp, dst_zp, beta, j, compensation);
        i += 4;
    }
    if (i < M)
        scalar_tail_kernel_col_major(&src[i * src_stride], &dst[N * dst_stride + i], M - i, N, src_stride, dst_stride,
                           src_scale, dst_scale, src_zp, dst_zp, beta, 0, compensation);
}

} // namespace tr
} // namespace ppc64
} // namespace cpu
} // namespace impl
} // namespace dnnl
