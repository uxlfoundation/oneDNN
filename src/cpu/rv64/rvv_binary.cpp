/******************************************************************************
* Copyright 2025
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
******************************************************************************/

#include <assert.h>
#include <memory>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/primitive_attr_postops.hpp"
#include "cpu/rv64/rvv_binary_kernels.hpp"

#include "cpu/rv64/rvv_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Data type dispatch for RVV binary
static inline void compute_binary_rvv(const alg_kind_t alg, const void *x,
        const void *y, void *dst, const int8_t *c, const dim_t len,
        const data_type_t dt) {
    switch (dt) {
        case data_type::f32:
            rvv_binary_apply_f32(alg, static_cast<const float *>(x),
                    static_cast<const float *>(y), static_cast<float *>(dst), c,
                    len);
            break;
        case data_type::f16:
            rvv_binary_apply_f16(alg, static_cast<const _Float16 *>(x),
                    static_cast<const _Float16 *>(y),
                    static_cast<_Float16 *>(dst), c, len);
            break;
        case data_type::s32:
            rvv_binary_apply_s32(alg, static_cast<const int32_t *>(x),
                    static_cast<const int32_t *>(y),
                    static_cast<int32_t *>(dst), c, len);
            break;
        case data_type::s8:
            rvv_binary_apply_s8(alg, static_cast<const int8_t *>(x),
                    static_cast<const int8_t *>(y), static_cast<int8_t *>(dst),
                    c, len);
            break;
        case data_type::u8:
            rvv_binary_apply_u8(alg, static_cast<const uint8_t *>(x),
                    static_cast<const uint8_t *>(y),
                    static_cast<uint8_t *>(dst), c, len);
            break;
        default: assert(!"Unsupported data type for RVV binary");
    }
}

template <data_type_t data_type>
status_t rvv_binary_t<data_type>::execute_binary(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;
    auto src0 = CTX_IN_MEM(
            const typename prec_traits_t<data_type>::type *, DNNL_ARG_SRC_0);
    auto src1 = CTX_IN_MEM(
            const typename prec_traits_t<data_type>::type *, DNNL_ARG_SRC_1);
    // src2 is optional (only for ternary ops like select); treat as s8 mask
    const int8_t *src2_s8 = nullptr;
    if (pd()->is_ternary_op()) {
        src2_s8 = CTX_IN_MEM(const int8_t *, DNNL_ARG_SRC_2);
    }
    auto dst = CTX_OUT_CLEAN_MEM(
            typename prec_traits_t<data_type>::type *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const memory_desc_wrapper src2_d(pd()->src_md(2));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto nelems = dst_d.nelems(true);

    const auto alg_kind = pd()->desc()->alg_kind;

    if (pd()->use_dense_) {
        const auto off0 = src0_d.offset0();
        const auto off1 = src1_d.offset0();
        const auto offd = dst_d.offset0();
        const auto off2 = pd()->is_ternary_op() ? src2_d.offset0() : 0;

        const auto *x_base = src0 + off0;
        const auto *y_base = src1 + off1;
        auto *d_base = dst + offd;
        const int8_t *c_base
                = pd()->is_ternary_op() ? (src2_s8 + off2) : nullptr;

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            const void *thr_x = static_cast<const void *>(x_base + start);
            const void *thr_y = static_cast<const void *>(y_base + start);
            void *thr_d = static_cast<void *>(d_base + start);
            const int8_t *thr_c = c_base ? (c_base + start) : nullptr;
            const dim_t len = end - start;

            compute_binary_rvv(alg_kind, thr_x, thr_y, thr_d, thr_c, len,
                    pd()->dst_md()->data_type);
        });

        return status::success;
    }

    if (pd()->use_nCspBc_padded_) {
        const blocking_desc_t &blk = dst_d.blocking_desc();
        const dim_t block = blk.inner_blks[0];

        const dim_t MB = dst_d.dims()[0];
        const dim_t C_PADDED = dst_d.padded_dims()[1] / block;
        const dim_t C = dst_d.dims()[1] / block;
        const dim_t tail = dst_d.dims()[1] % block;
        const dim_t D = (dst_d.ndims() > 4 ? dst_d.dims()[2] : 1);
        const dim_t H
                = (dst_d.ndims() > 3 ? dst_d.dims()[dst_d.ndims() - 2] : 1);
        const dim_t W
                = (dst_d.ndims() > 2 ? dst_d.dims()[dst_d.ndims() - 1] : 1);
        const dim_t SP = D * H * W;

        parallel_nd(MB, C_PADDED, SP, [&](dim_t n, dim_t c, dim_t sp) {
            // Decompose sp into D/H/W indices for proper broadcasting
            dim_t dd = 0, hh = 0, ww = 0;
            if (dst_d.ndims() > 4) {
                const dim_t HW = H * W;
                dd = sp / HW;
                const dim_t hw_rem = sp % HW;
                hh = hw_rem / W;
                ww = hw_rem % W;
            } else if (dst_d.ndims() > 3) { // 4D: NCHW
                hh = sp / W;
                ww = sp % W;
            } else { // 3D or lower: treat as single spatial dim
                ww = sp;
            }

            // Starting channel index of the current block
            const dim_t c_start = c * block;

            // Helper to fill logical positions per md with broadcasting
            auto fill_pos = [&](const memory_desc_wrapper &md, dims_t &pos) {
                for (int k = 0; k < md.ndims(); ++k)
                    pos[k] = 0;
                const int nd = md.ndims();
                // N
                pos[0] = (md.dims()[0] == 1) ? 0 : n;
                // C (logical channel index)
                pos[1] = (md.dims()[1] == 1) ? 0 : c_start;
                if (nd > 4) {
                    // D, H, W
                    const dim_t Dm = md.dims()[2];
                    const dim_t Hm = md.dims()[nd - 2];
                    const dim_t Wm = md.dims()[nd - 1];
                    pos[2] = (Dm == 1) ? 0 : dd;
                    pos[nd - 2] = (Hm == 1) ? 0 : hh;
                    pos[nd - 1] = (Wm == 1) ? 0 : ww;
                } else if (nd > 3) {
                    // H, W
                    const dim_t Hm = md.dims()[nd - 2];
                    const dim_t Wm = md.dims()[nd - 1];
                    pos[nd - 2] = (Hm == 1) ? 0 : hh;
                    pos[nd - 1] = (Wm == 1) ? 0 : ww;
                } else if (nd > 2) {
                    // Single spatial (W)
                    const dim_t Wm = md.dims()[nd - 1];
                    pos[nd - 1] = (Wm == 1) ? 0 : ww;
                }
            };

            // Compute per-tensor offsets (include offset0())
            dims_t pos;
            fill_pos(dst_d, pos);
            const dim_t d_off = dst_d.off_v(pos, false);
            fill_pos(src0_d, pos);
            const dim_t x_off = src0_d.off_v(pos, false);
            fill_pos(src1_d, pos);
            const dim_t y_off = src1_d.off_v(pos, false);
            dim_t c_off = 0;
            if (pd()->is_ternary_op()) {
                fill_pos(src2_d, pos);
                c_off = src2_d.off_v(pos, false);
            }

            const dim_t len = (c < C) ? block : ((c == C) ? tail : (dim_t)0);
            if (len == 0) return;

            using data_t = typename prec_traits_t<data_type>::type;
            const data_t *x_ptr = src0 + x_off;
            const data_t *y_ptr = src1 + y_off;
            data_t *d_ptr = dst + d_off;
            const int8_t *c_ptr
                    = pd()->is_ternary_op() ? (src2_s8 + c_off) : nullptr;

            // Replicate along C when the source has C==1 to avoid reading padded garbage
            // in blocked layout.
            const bool x_c_bcast = src0_d.dims()[1] == 1;
            const bool y_c_bcast = src1_d.dims()[1] == 1;
            const bool c_c_bcast
                    = pd()->is_ternary_op() && src2_d.dims()[1] == 1;

            // Small stack buffers for typical block sizes; fall back to heap if needed.
            data_t x_tmp_stack[64];
            data_t y_tmp_stack[64];
            int8_t c_tmp_stack[64];

            std::unique_ptr<data_t[]> x_tmp_heap;
            std::unique_ptr<data_t[]> y_tmp_heap;
            std::unique_ptr<int8_t[]> c_tmp_heap;

            if (x_c_bcast) {
                data_t *buf = (len <= (dim_t)64)
                        ? x_tmp_stack
                        : (x_tmp_heap.reset(new data_t[len]), x_tmp_heap.get());
                const data_t val = *x_ptr;
                for (dim_t i = 0; i < len; ++i)
                    buf[i] = val;
                x_ptr = buf;
            }
            if (y_c_bcast) {
                data_t *buf = (len <= (dim_t)64)
                        ? y_tmp_stack
                        : (y_tmp_heap.reset(new data_t[len]), y_tmp_heap.get());
                const data_t val = *y_ptr;
                for (dim_t i = 0; i < len; ++i)
                    buf[i] = val;
                y_ptr = buf;
            }
            if (c_ptr && c_c_bcast) {
                int8_t *buf = (len <= (dim_t)64)
                        ? c_tmp_stack
                        : (c_tmp_heap.reset(new int8_t[len]), c_tmp_heap.get());
                const int8_t val = *c_ptr;
                for (dim_t i = 0; i < len; ++i)
                    buf[i] = val;
                c_ptr = buf;
            }

            compute_binary_rvv(alg_kind, static_cast<const void *>(x_ptr),
                    static_cast<const void *>(y_ptr),
                    static_cast<void *>(d_ptr), c_ptr, len,
                    pd()->dst_md()->data_type);
        });

        return status::success;
    }

    return status::unimplemented;
}

// Explicit template instantiations for forward RVV binary
template status_t rvv_binary_t<data_type::f32>::execute_binary(
        const exec_ctx_t &) const;
template status_t rvv_binary_t<data_type::f16>::execute_binary(
        const exec_ctx_t &) const;
template status_t rvv_binary_t<data_type::s32>::execute_binary(
        const exec_ctx_t &) const;
template status_t rvv_binary_t<data_type::s8>::execute_binary(
        const exec_ctx_t &) const;
template status_t rvv_binary_t<data_type::u8>::execute_binary(
        const exec_ctx_t &) const;

template struct rvv_binary_t<data_type::f32>;
template struct rvv_binary_t<data_type::f16>;
template struct rvv_binary_t<data_type::s32>;
template struct rvv_binary_t<data_type::s8>;
template struct rvv_binary_t<data_type::u8>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl