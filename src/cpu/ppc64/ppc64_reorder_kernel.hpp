#ifndef CPU_PPC64_REORDER_KERNEL_HPP
#define CPU_PPC64_REORDER_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include "cpu/ppc64/ppc64_uni_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace ppc64 {
namespace tr {

enum class layout_kind_t {
    row_major,
    col_major,
    blocked_transpose,
    fallback_scalar
};

struct reorder_s8_s8_kernel_t : public kernel_t {
    reorder_s8_s8_kernel_t(const desc_t &desc);
    void operator()(const call_param_t *p) const override;
    void operator()(const tail_call_param_t *p) const override;
    static bool applicable(const prb_t &prb);
    status_t create_kernel() override { return status::success; }

private:
    layout_kind_t detect_layout(const node_t &n0, const node_t &n1) const;
    void copy_nd(const int8_t *src, int8_t *dst, const node_t *nodes, int ndims,
            float src_scale, float dst_scale, int src_zp, int dst_zp,
            float beta, int32_t *compensation) const;
    void copy_2d(const int8_t *src, int8_t *dst, const node_t &n0,
            const node_t &n1, float src_scale, float dst_scale, int src_zp,
            int dst_zp, float beta, int32_t *compensation) const;
    void copy_1d(const int8_t *src, int8_t *dst, const node_t &n0,
            float src_scale, float dst_scale, int src_zp, int dst_zp,
            float beta, int32_t *compensation) const;
    // Block kernels
    static void block_kernel_16x16_row_major(const int8_t *src, int8_t *dst,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int comp_offset,
            int32_t *compensation);
    static void block_kernel_8x8_row_major(const int8_t *src, int8_t *dst,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int comp_offset,
            int32_t *compensation);
    static void block_kernel_4x4_row_major(const int8_t *src, int8_t *dst,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int comp_offset,
            int32_t *compensation);
    static void block_kernel_16x16_col_major(const int8_t *src, int8_t *dst,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int comp_offset,
            int32_t *compensation);
    static void block_kernel_8x8_col_major(const int8_t *src, int8_t *dst,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int comp_offset,
            int32_t *compensation);
    static void block_kernel_4x4_col_major(const int8_t *src, int8_t *dst,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int comp_offset,
            int32_t *compensation);
    static void scalar_tail_kernel(const int8_t *src, int8_t *dst, int M, int N,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int comp_offset,
            int32_t *compensation);
    static void scalar_tail_kernel_row_major(const int8_t *src, int8_t *dst,
            int M, int N, int src_stride, int dst_stride, float src_scale,
            float dst_scale, int src_zp, int dst_zp, float beta,
            int comp_offset, int32_t *compensation);
    static void scalar_tail_kernel_col_major(const int8_t *src, int8_t *dst,
            int M, int N, int src_stride, int dst_stride, float src_scale,
            float dst_scale, int src_zp, int dst_zp, float beta,
            int comp_offset, int32_t *compensation);
    // Tiling helpers
    void reorder_row_major_kernel(const int8_t *src, int8_t *dst, int M, int N,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int32_t *compensation) const;
    void reorder_col_major_kernel(const int8_t *src, int8_t *dst, int M, int N,
            int src_stride, int dst_stride, float src_scale, float dst_scale,
            int src_zp, int dst_zp, float beta, int32_t *compensation) const;
};

} // namespace ppc64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
