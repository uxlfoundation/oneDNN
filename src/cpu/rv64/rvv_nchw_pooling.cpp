/******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2023-2025 KNS Group LLC (YADRO)
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

#if defined(DNNL_RISCV_USE_RVV_INTRINSICS)
#include <riscv_vector.h>
#endif

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "cpu/rv64/jit_rvv_pooling_kernel.hpp"
#include "cpu/rv64/rvv_nchw_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

// JIT NCHW pooling: all f32 paths use JIT-generated code

using kernel_t = jit_rvv_pooling_fwd_kernel_t;

struct init_scale_t {
    float init_val;
    float scale_val;
};

init_scale_t compute_init_scale(kernel_t::alg_t alg, int id_start, int id_end,
        int ih_start, int ih_end, int iw_start, int iw_end, dim_t kerD,
        dim_t kerH, dim_t kerW) {
    if (alg == kernel_t::alg_t::max_pool) {
        return {-FLT_MAX, 0.0f};
    } else if (alg == kernel_t::alg_t::avg_include) {
        return {0.0f, 1.0f / (float)(kerD * kerH * kerW)};
    } else {
        const int count = std::max(0, id_end - id_start)
                * std::max(0, ih_end - ih_start)
                * std::max(0, iw_end - iw_start);
        return {0.0f, (count > 0) ? 1.0f / (float)count : 0.0f};
    }
}

template <kernel_t::alg_t alg>
static float compute_scalar_pool(const float *src, int id_start, int id_end,
        int ih_start, int ih_end, int iw_start, int iw_end, dim_t inW,
        dim_t inH, dim_t kerD, dim_t kerH, dim_t kerW, bool with_relu,
        float relu_alpha) {
    float acc = (alg == kernel_t::alg_t::max_pool) ? -FLT_MAX : 0.0f;
    for (int id = id_start; id < id_end; id++)
        for (int ih = ih_start; ih < ih_end; ih++)
            for (int iw = iw_start; iw < iw_end; iw++) {
                const float val = src[id * inH * inW + ih * inW + iw];
                if (alg == kernel_t::alg_t::max_pool) {
                    if (val > acc) acc = val;
                } else {
                    acc += val;
                }
            }

    if (alg == kernel_t::alg_t::avg_include) {
        acc *= 1.0f / (float)(kerD * kerH * kerW);
    } else if (alg == kernel_t::alg_t::avg_exclude) {
        const int count = std::max(0, id_end - id_start)
                * std::max(0, ih_end - ih_start)
                * std::max(0, iw_end - iw_start);
        acc = (count > 0) ? acc * (1.0f / (float)count) : 0.0f;
    }

    if (with_relu) {
        if (relu_alpha == 0.0f) {
            acc = std::max(acc, 0.0f);
        } else {
            acc = (acc > 0.0f) ? acc : acc * relu_alpha;
        }
    }

    return acc;
}

template <kernel_t::alg_t alg>
void Pooling_f32_jit_nchw(const float *src, float *dst, dim_t batch,
        dim_t channels, dim_t outD, dim_t outH, dim_t outW, dim_t inD,
        dim_t inH, dim_t inW, dim_t kerD, dim_t kerH, dim_t kerW, dim_t strideD,
        dim_t strideH, dim_t strideW, dim_t padF, dim_t padT, dim_t padL,
        bool with_relu, float relu_alpha) {

    static const kernel_t kernel(alg);

    const size_t spatial_size = (size_t)inD * inH * inW;
    const size_t dst_spatial_size = (size_t)outD * outH * outW;

    // Interior OW range: positions with full kernel window in W (no truncation).
    // ow*SW - padL >= 0 AND ow*SW - padL + kerW <= inW
    //   => ow >= ceil(padL/SW) AND ow <= (inW - kerW + padL) / SW
    const dim_t int_lo = std::max((padL + strideW - 1) / strideW, (dim_t)0);
    const dim_t int_hi = std::min(
            std::max((inW - kerW + padL) / strideW + 1, int_lo), outW);

    parallel_nd(batch, channels, outD, outH,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh) {
        const float *src_ch = src + ((size_t)mb * channels + c) * spatial_size;
        float *dst_oh = dst + ((size_t)mb * channels + c) * dst_spatial_size
                + od * outH * outW + oh * outW;

        const int id_start = std::max((int)(od * strideD - padF), 0);
        const int id_end
                = std::min((int)(od * strideD - padF + kerD), (int)inD);
        const int ih_start = std::max((int)(oh * strideH - padT), 0);
        const int ih_end
                = std::min((int)(oh * strideH - padT + kerH), (int)inH);

        // Left boundary OW positions — scalar (avoids JIT call overhead)
        for (dim_t ow = 0; ow < int_lo && ow < outW; ow++) {
            const int iw_s = std::max((int)(ow * strideW - padL), 0);
            const int iw_e
                    = std::min((int)(ow * strideW - padL + kerW), (int)inW);
            dst_oh[ow] = compute_scalar_pool<alg>(src_ch, id_start, id_end,
                    ih_start, ih_end, iw_s, iw_e, inW, inH, kerD, kerH, kerW,
                    with_relu, relu_alpha);
        }

        // Interior OW positions — vectorized along OW
        if (int_lo < int_hi) {

            // Interior has full kernel window in W, so iw range = [0, kerW)
            auto iv = compute_init_scale(alg, id_start, id_end, ih_start,
                    ih_end, 0, (int)kerW, kerD, kerH, kerW);

            kernel_t::call_params_t p;
            p.src = src_ch + (int_lo * strideW - padL);
            p.dst = dst_oh + int_lo;
            // Repurpose 'channels' for OW count: the JIT kernel's channel
            // loop iterates over interior OW positions with
            // src_vec_byte_stride advancing between adjacent output-width
            // positions.
            p.channels = int_hi - int_lo;
            p.id_start = id_start;
            p.ih_start = ih_start;
            p.iw_start = 0;
            p.id_end = id_end;
            p.ih_end = ih_end;
            p.iw_end = kerW;
            p.inW_stride = inW;
            p.inD_stride = inH * inW;
            p.w_spatial_byte_stride = (dim_t)sizeof(float);
            p.init_val = iv.init_val;
            p.scale_val = iv.scale_val;
            p.relu_alpha = relu_alpha;
            p.with_relu = with_relu;
            p.src_vec_byte_stride = strideW * (dim_t)sizeof(float);
            p.dst_vec_byte_stride = (dim_t)sizeof(float);
            kernel(&p);
        }

        // Right boundary OW positions — scalar (avoids JIT call overhead)
        for (dim_t ow = int_hi; ow < outW; ow++) {
            const int iw_s = std::max((int)(ow * strideW - padL), 0);
            const int iw_e
                    = std::min((int)(ow * strideW - padL + kerW), (int)inW);
            dst_oh[ow] = compute_scalar_pool<alg>(src_ch, id_start, id_end,
                    ih_start, ih_end, iw_s, iw_e, inW, inH, kerD, kerH, kerW,
                    with_relu, relu_alpha);
        }
    });
}

template <kernel_t::alg_t alg>
void Pooling_f32_jit_nchw_ow1(const float *src, float *dst, dim_t batch,
        dim_t channels, dim_t outD, dim_t outH, dim_t outW, dim_t inD,
        dim_t inH, dim_t inW, dim_t kerD, dim_t kerH, dim_t kerW, dim_t strideD,
        dim_t strideH, dim_t strideW, dim_t padF, dim_t padT, dim_t padL,
        bool with_relu, float relu_alpha) {

    static const kernel_t kernel(alg);

    const size_t spatial_size = (size_t)inD * inH * inW;
    const size_t dst_spatial_size = (size_t)outD * outH * outW;

    parallel_nd(batch, outD, outH, [&](dim_t mb, dim_t od, dim_t oh) {
        const int id_start = std::max((int)(od * strideD - padF), 0);
        const int id_end
                = std::min((int)(od * strideD - padF + kerD), (int)inD);
        const int ih_start = std::max((int)(oh * strideH - padT), 0);
        const int ih_end
                = std::min((int)(oh * strideH - padT + kerH), (int)inH);
        const int iw_start = std::max(-(int)padL, 0);
        const int iw_end = std::min(-(int)padL + (int)kerW, (int)inW);

        auto iv = compute_init_scale(alg, id_start, id_end, ih_start, ih_end,
                iw_start, iw_end, kerD, kerH, kerW);

        // src_base points to channel 0 at spatial offset (id_start, ih_start,
        // iw_start). The kernel's spatial loops start from 0 and iterate within
        // the window, so offsets are baked into the base pointer.
        const float *src_base = src + (size_t)mb * channels * spatial_size
                + (size_t)id_start * inH * inW + (size_t)ih_start * inW
                + iw_start;
        float *dst_base = dst + (size_t)mb * channels * dst_spatial_size
                + od * outH + oh;

        kernel_t::call_params_t p;
        p.src = src_base;
        p.dst = dst_base;
        p.channels = channels;
        p.id_start = 0;
        p.ih_start = 0;
        p.iw_start = 0;
        p.id_end = id_end - id_start;
        p.ih_end = ih_end - ih_start;
        p.iw_end = iw_end - iw_start;
        p.inW_stride = inW;
        p.inD_stride = inH * inW;
        p.w_spatial_byte_stride = (dim_t)sizeof(float);
        p.init_val = iv.init_val;
        p.scale_val = iv.scale_val;
        p.relu_alpha = relu_alpha;
        p.with_relu = with_relu;
        p.src_vec_byte_stride = (dim_t)(spatial_size * sizeof(float));
        p.dst_vec_byte_stride = (dim_t)(dst_spatial_size * sizeof(float));
        kernel(&p);
    });
}

// === Intrinsics-based fallback for low-channel OW>1 shapes ===
// The JIT kernel has per-call overhead (callee-save/restore, parameter loads)
// that is poorly amortized when channel count is low. For C < 288, the
// intrinsics path avoids this overhead and matches the original baseline.

#if defined(DNNL_RISCV_USE_RVV_INTRINSICS)
static void MaxPooling_intrin(const float *src, float *dst, dim_t batch,
        dim_t channels, dim_t outD, dim_t outH, dim_t outW, dim_t inD,
        dim_t inH, dim_t inW, dim_t kerD, dim_t kerH, dim_t kerW, dim_t strideD,
        dim_t strideH, dim_t strideW, dim_t padFront, dim_t padTop,
        dim_t padLeft) {

    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_offset = (size_t)outW * outH * outD * channels * mb
                + (size_t)outW * outH * outD * c + (size_t)outW * outH * od
                + (size_t)outW * oh + (size_t)ow;
        const auto src_offset
                = ((size_t)inW * inH * inD) * ((size_t)channels * mb + c);
        const auto local_src = &src[src_offset];
        const auto IWH = (size_t)inW * inH;

        int od_offset = od * strideD - padFront;
        int oh_offset = oh * strideH - padTop;
        int ow_offset = ow * strideW - padLeft;
        int iw_start = std::max(ow_offset, 0);
        int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = -FLT_MAX;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-FLT_MAX, cycleLength);

        for (int id = std::max(od_offset, 0);
                id < std::min(od_offset + (int)kerD, (int)inD); id++)
            for (int ih = std::max(oh_offset, 0);
                    ih < std::min(oh_offset + (int)kerH, (int)inH); ih++) {
                const auto local_src_offset
                        = IWH * id + (size_t)inW * ih + std::max(ow_offset, 0);

                size_t iw = 0;
                for (; iw + cycleLength <= size; iw += cycleLength) {
                    vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(
                            &local_src[local_src_offset + iw], cycleLength);
                    vmax = __riscv_vfmax_vv_f32m1(vsrc, vmax, cycleLength);
                }

                size_t tailLength = __riscv_vsetvl_e32m1(size - iw);
                {
                    vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(
                            &local_src[local_src_offset + iw], tailLength);
                    vmax = __riscv_vfmax_vv_f32m1(vsrc, vmax, tailLength);
                }
            }

        vfloat32m1_t min_scalar = __riscv_vfmv_v_f_f32m1(-FLT_MAX, 1);
        cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vred_res;
        vred_res = __riscv_vfredmax_vs_f32m1_f32m1(
                vmax, min_scalar, cycleLength);
        __riscv_vse32_v_f32m1(&dst[dst_offset], vred_res, 1);
    });
}

static void AvgPoolingIncludePadding_intrin(const float *src, float *dst,
        dim_t batch, dim_t channels, dim_t outD, dim_t outH, dim_t outW,
        dim_t inD, dim_t inH, dim_t inW, dim_t kerD, dim_t kerH, dim_t kerW,
        dim_t strideD, dim_t strideH, dim_t strideW, dim_t padFront,
        dim_t padTop, dim_t padLeft) {

    const float kernel_volume = (float)(kerD * kerH * kerW);

    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_offset = (size_t)outW * outH * outD * channels * mb
                + (size_t)outW * outH * outD * c + (size_t)outW * outH * od
                + (size_t)outW * oh + (size_t)ow;
        const auto src_offset
                = ((size_t)inW * inH * inD) * ((size_t)channels * mb + c);
        const auto local_src = &src[src_offset];
        const auto IWH = (size_t)inW * inH;

        int od_offset = od * strideD - padFront;
        int oh_offset = oh * strideH - padTop;
        int ow_offset = ow * strideW - padLeft;
        int iw_start = std::max(ow_offset, 0);
        int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = 0.0f;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, cycleLength);

        for (int id = std::max(od_offset, 0);
                id < std::min(od_offset + (int)kerD, (int)inD); id++)
            for (int ih = std::max(oh_offset, 0);
                    ih < std::min(oh_offset + (int)kerH, (int)inH); ih++) {
                const size_t local_src_offset
                        = IWH * id + (size_t)inW * ih + iw_start;

                size_t iw = 0;
                for (; iw + cycleLength <= size; iw += cycleLength) {
                    vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(
                            &local_src[local_src_offset + iw], cycleLength);
                    vsum = __riscv_vfadd_vv_f32m1(vsum, vsrc, cycleLength);
                }

                size_t tailLength = __riscv_vsetvl_e32m1(size - iw);
                {
                    vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(
                            &local_src[local_src_offset + iw], tailLength);
                    vsum = __riscv_vfadd_vv_f32m1(vsum, vsrc, tailLength);
                }
            }

        vfloat32m1_t zero_scalar = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vred_res;
        vred_res = __riscv_vfredusum_vs_f32m1_f32m1(
                vsum, zero_scalar, cycleLength);

        float red_res;
        __riscv_vse32_v_f32m1(&red_res, vred_res, 1);
        dst[dst_offset] = red_res / kernel_volume;
    });
}

static void AvgPoolingExcludePadding_intrin(const float *src, float *dst,
        dim_t batch, dim_t channels, dim_t outD, dim_t outH, dim_t outW,
        dim_t inD, dim_t inH, dim_t inW, dim_t kerD, dim_t kerH, dim_t kerW,
        dim_t strideD, dim_t strideH, dim_t strideW, dim_t padFront,
        dim_t padTop, dim_t padLeft) {

    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_offset = (size_t)outW * outH * outD * channels * mb
                + (size_t)outW * outH * outD * c + (size_t)outW * outH * od
                + (size_t)outW * oh + (size_t)ow;
        const auto src_offset
                = ((size_t)inW * inH * inD) * ((size_t)channels * mb + c);
        const auto local_src = &src[src_offset];
        const auto IWH = (size_t)inW * inH;

        int od_offset = od * strideD - padFront;
        int oh_offset = oh * strideH - padTop;
        int ow_offset = ow * strideW - padLeft;
        int iw_start = std::max(ow_offset, 0);
        int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = 0.0f;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, cycleLength);
        size_t count = 0;

        for (int id = od_offset; id < od_offset + (int)kerD; id++) {
            if (id < 0 || id >= (int)inD) continue;
            for (int ih = oh_offset; ih < oh_offset + (int)kerH; ih++) {
                if (ih < 0 || ih >= (int)inH) continue;
                if (iw_start >= iw_end) continue;

                const size_t local_src_offset
                        = IWH * id + (size_t)inW * ih + iw_start;
                size_t iw = 0;

                for (; iw + cycleLength <= size; iw += cycleLength) {
                    vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(
                            &local_src[local_src_offset + iw], cycleLength);
                    vsum = __riscv_vfadd_vv_f32m1(vsum, vsrc, cycleLength);
                }

                size_t tailLength = __riscv_vsetvl_e32m1(size - iw);
                {
                    vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(
                            &local_src[local_src_offset + iw], tailLength);
                    vsum = __riscv_vfadd_vv_f32m1(vsum, vsrc, tailLength);
                }

                count += size;
            }
        }

        if (count == 0) {
            dst[dst_offset] = 0.0f;
            return;
        }

        vfloat32m1_t zero_scalar = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vred_res;
        vred_res = __riscv_vfredusum_vs_f32m1_f32m1(
                vsum, zero_scalar, cycleLength);

        float red_res;
        __riscv_vse32_v_f32m1(&red_res, vred_res, 1);
        dst[dst_offset] = red_res / (float)count;
    });
}
#endif // DNNL_RISCV_USE_RVV_INTRINSICS

#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
#include <riscv_vector.h>

static void MaxPooling_f16(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, const dim_t batch, const dim_t channels,
        const dim_t outD, const dim_t outH, const dim_t outW, const dim_t inD,
        const dim_t inH, const dim_t inW, const dim_t kerD, const dim_t kerH,
        const dim_t kerW, const dim_t strideD, const dim_t strideH,
        const dim_t strideW, const dim_t padFront, const dim_t padTop,
        const dim_t padLeft) {

    const float f16_lowest
            = (float)nstl::numeric_limits<dnnl::impl::float16_t>::lowest();

    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_offset = (size_t)outW * outH * outD * channels * mb
                + (size_t)outW * outH * outD * c + (size_t)outW * outH * od
                + (size_t)outW * oh + (size_t)ow;
        const auto src_offset
                = ((size_t)inW * inH * inD) * ((size_t)channels * mb + c);
        const auto local_src = &src[src_offset];
        const auto IWH = (size_t)inW * inH;

        int od_offset = od * strideD - padFront;
        int oh_offset = oh * strideH - padTop;
        int ow_offset = ow * strideW - padLeft;
        int iw_start = std::max(ow_offset, 0);
        int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = (dnnl::impl::float16_t)f16_lowest;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t vl = __riscv_vsetvl_e16m1(size);
        vfloat16m1_t vmax = __riscv_vfmv_v_f_f16m1((_Float16)f16_lowest, vl);

        for (int id = std::max(od_offset, 0);
                id < std::min(od_offset + (int)kerD, (int)inD); id++)
            for (int ih = std::max(oh_offset, 0);
                    ih < std::min(oh_offset + (int)kerH, (int)inH); ih++) {
                const auto local_src_offset
                        = IWH * id + (size_t)inW * ih + iw_start;
                size_t iw = 0;
                while (iw < size) {
                    size_t vli = __riscv_vsetvl_e16m1(size - iw);
                    vfloat16m1_t vsrc = __riscv_vle16_v_f16m1(
                            (const _Float16 *)&local_src[local_src_offset + iw],
                            vli);
                    vmax = __riscv_vfmax_vv_f16m1(vmax, vsrc, vli);
                    iw += vli;
                }
            }

        vfloat16m1_t min_scalar
                = __riscv_vfmv_v_f_f16m1((_Float16)f16_lowest, 1);
        vfloat16m1_t vred_res
                = __riscv_vfredmax_vs_f16m1_f16m1(vmax, min_scalar, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&dst[dst_offset], vred_res, 1);
    });
}

static void AvgPoolingIncludePadding_f16(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, const dim_t batch, const dim_t channels,
        const dim_t outD, const dim_t outH, const dim_t outW, const dim_t inD,
        const dim_t inH, const dim_t inW, const dim_t kerD, const dim_t kerH,
        const dim_t kerW, const dim_t strideD, const dim_t strideH,
        const dim_t strideW, const dim_t padFront, const dim_t padTop,
        const dim_t padLeft) {

    const float kernel_volume = (float)(kerD * kerH * kerW);

    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_offset = (size_t)outW * outH * outD * channels * mb
                + (size_t)outW * outH * outD * c + (size_t)outW * outH * od
                + (size_t)outW * oh + (size_t)ow;
        const auto src_offset
                = ((size_t)inW * inH * inD) * ((size_t)channels * mb + c);
        const auto local_src = &src[src_offset];
        const auto IWH = (size_t)inW * inH;

        int od_offset = od * strideD - padFront;
        int oh_offset = oh * strideH - padTop;
        int ow_offset = ow * strideW - padLeft;
        int iw_start = std::max(ow_offset, 0);
        int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = (dnnl::impl::float16_t)0.0f;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t vl = __riscv_vsetvl_e16m1(size);
        vfloat32m2_t vsum_f32 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (int id = std::max(od_offset, 0);
                id < std::min(od_offset + (int)kerD, (int)inD); id++)
            for (int ih = std::max(oh_offset, 0);
                    ih < std::min(oh_offset + (int)kerH, (int)inH); ih++) {
                const size_t local_src_offset
                        = IWH * id + (size_t)inW * ih + iw_start;
                size_t iw = 0;
                while (iw < size) {
                    size_t vli = __riscv_vsetvl_e16m1(size - iw);
                    vfloat16m1_t vsrc_f16 = __riscv_vle16_v_f16m1(
                            (const _Float16 *)&local_src[local_src_offset + iw],
                            vli);
                    vfloat32m2_t vsrc_f32
                            = __riscv_vfwcvt_f_f_v_f32m2(vsrc_f16, vli);
                    vsum_f32 = __riscv_vfadd_vv_f32m2(vsum_f32, vsrc_f32, vli);
                    iw += vli;
                }
            }

        vfloat32m1_t zero_scalar = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t vred_res
                = __riscv_vfredusum_vs_f32m2_f32m1(vsum_f32, zero_scalar, vl);
        float red_res;
        __riscv_vse32_v_f32m1(&red_res, vred_res, 1);
        dst[dst_offset] = (dnnl::impl::float16_t)(red_res / kernel_volume);
    });
}

static void AvgPoolingExcludePadding_f16(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, const dim_t batch, const dim_t channels,
        const dim_t outD, const dim_t outH, const dim_t outW, const dim_t inD,
        const dim_t inH, const dim_t inW, const dim_t kerD, const dim_t kerH,
        const dim_t kerW, const dim_t strideD, const dim_t strideH,
        const dim_t strideW, const dim_t padFront, const dim_t padTop,
        const dim_t padLeft) {

    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_offset = (size_t)outW * outH * outD * channels * mb
                + (size_t)outW * outH * outD * c + (size_t)outW * outH * od
                + (size_t)outW * oh + (size_t)ow;
        const auto src_offset
                = ((size_t)inW * inH * inD) * ((size_t)channels * mb + c);
        const auto local_src = &src[src_offset];
        const auto IWH = (size_t)inW * inH;

        int od_offset = od * strideD - padFront;
        int oh_offset = oh * strideH - padTop;
        int ow_offset = ow * strideW - padLeft;
        int iw_start = std::max(ow_offset, 0);
        int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = (dnnl::impl::float16_t)0.0f;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t vl = __riscv_vsetvl_e16m1(size);
        vfloat32m2_t vsum_f32 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        size_t count = 0;

        for (int id = od_offset; id < od_offset + (int)kerD; id++) {
            if (id < 0 || id >= inD) continue;
            for (int ih = oh_offset; ih < oh_offset + (int)kerH; ih++) {
                if (ih < 0 || ih >= inH) continue;

                const size_t local_src_offset
                        = IWH * id + (size_t)inW * ih + iw_start;
                size_t iw = 0;
                while (iw < size) {
                    size_t vli = __riscv_vsetvl_e16m1(size - iw);
                    vfloat16m1_t vsrc_f16 = __riscv_vle16_v_f16m1(
                            (const _Float16 *)&local_src[local_src_offset + iw],
                            vli);
                    vfloat32m2_t vsrc_f32
                            = __riscv_vfwcvt_f_f_v_f32m2(vsrc_f16, vli);
                    vsum_f32 = __riscv_vfadd_vv_f32m2(vsum_f32, vsrc_f32, vli);
                    iw += vli;
                }
                count += size;
            }
        }

        if (count == 0) {
            dst[dst_offset] = (dnnl::impl::float16_t)0.0f;
            return;
        }

        vfloat32m1_t zero_scalar = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t vred_res
                = __riscv_vfredusum_vs_f32m2_f32m1(vsum_f32, zero_scalar, vl);
        float red_res;
        __riscv_vse32_v_f32m1(&red_res, vred_res, 1);
        dst[dst_offset] = (dnnl::impl::float16_t)(red_res / (float)count);
    });
}
#endif

} // namespace

riscv_nchw_pooling_fwd_t::riscv_nchw_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

status_t riscv_nchw_pooling_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src_raw = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst_raw = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto dt = src_d.data_type();
    const size_t dt_size = types::data_type_size(dt);

    src_raw = (const uint8_t *)src_raw + src_d.off_l(0) * dt_size;
    dst_raw = (uint8_t *)dst_raw + dst_d.off_l(0) * dt_size;

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();

    const auto alg = pd()->desc()->alg_kind;
    const bool is_max_pool = alg == alg_kind::pooling_max;
    const bool is_avg_pool_include
            = alg == alg_kind::pooling_avg_include_padding;
    const bool is_avg_pool_exclude
            = alg == alg_kind::pooling_avg_exclude_padding;

    if (dt == data_type::f32) {
        const float *src = static_cast<const float *>(src_raw);
        float *dst = static_cast<float *>(dst_raw);

        rvv_postops_t postops_handler(pd()->attr()->post_ops_);
        const bool with_relu = postops_handler.is_relu_postop();
        const float relu_alpha = postops_handler.relu_alpha();

        // JIT OW-vectorized path has per-call overhead (callee-save/restore,
        // parameter loads) that is poorly amortized when channel count is low.
        // Use intrinsics fallback for C < 288 to avoid 10-27% regression on
        // low-channel, large-spatial shapes.
        static constexpr dim_t jit_ow_min_channels = 288;
        const bool use_jit_ow = (OW > 1) && (C >= jit_ow_min_channels);
        bool relu_fused_in_kernel = false;

        using kalg = kernel_t::alg_t;
        if (use_jit_ow) {
            // OW vectorization (any stride), high-C shapes only
            relu_fused_in_kernel = with_relu;
            if (is_max_pool) {
                Pooling_f32_jit_nchw<kalg::max_pool>(src, dst, MB, C, OD, OH,
                        OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else if (is_avg_pool_exclude) {
                Pooling_f32_jit_nchw<kalg::avg_exclude>(src, dst, MB, C, OD, OH,
                        OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else if (is_avg_pool_include) {
                Pooling_f32_jit_nchw<kalg::avg_include>(src, dst, MB, C, OD, OH,
                        OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else {
                return status::unimplemented;
            }
        } else if (OW == 1) {
            // OW=1: channel vectorization (no threshold needed)
            relu_fused_in_kernel = with_relu;
            if (is_max_pool) {
                Pooling_f32_jit_nchw_ow1<kalg::max_pool>(src, dst, MB, C, OD,
                        OH, OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else if (is_avg_pool_exclude) {
                Pooling_f32_jit_nchw_ow1<kalg::avg_exclude>(src, dst, MB, C, OD,
                        OH, OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else if (is_avg_pool_include) {
                Pooling_f32_jit_nchw_ow1<kalg::avg_include>(src, dst, MB, C, OD,
                        OH, OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else {
                return status::unimplemented;
            }
        } else {
            // Low-C OW>1: intrinsics fallback to avoid JIT overhead regression
#if defined(DNNL_RISCV_USE_RVV_INTRINSICS)
            if (is_max_pool) {
                MaxPooling_intrin(src, dst, MB, C, OD, OH, OW, ID, IH, IW, KD,
                        KH, KW, SD, SH, SW, padF, padT, padL);
            } else if (is_avg_pool_include) {
                AvgPoolingIncludePadding_intrin(src, dst, MB, C, OD, OH, OW, ID,
                        IH, IW, KD, KH, KW, SD, SH, SW, padF, padT, padL);
            } else if (is_avg_pool_exclude) {
                AvgPoolingExcludePadding_intrin(src, dst, MB, C, OD, OH, OW, ID,
                        IH, IW, KD, KH, KW, SD, SH, SW, padF, padT, padL);
            } else {
                return status::unimplemented;
            }
#else
            // No intrinsics available, use JIT path despite potential regression
            relu_fused_in_kernel = with_relu;
            if (is_max_pool) {
                Pooling_f32_jit_nchw<kalg::max_pool>(src, dst, MB, C, OD, OH,
                        OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else if (is_avg_pool_exclude) {
                Pooling_f32_jit_nchw<kalg::avg_exclude>(src, dst, MB, C, OD, OH,
                        OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else if (is_avg_pool_include) {
                Pooling_f32_jit_nchw<kalg::avg_include>(src, dst, MB, C, OD, OH,
                        OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT,
                        padL, with_relu, relu_alpha);
            } else {
                return status::unimplemented;
            }
#endif
        }

        // Post-ops: execute all post-ops for the intrinsics fallback path.
        // For the JIT path, ReLU was fused in the kernel; if the post-op is
        // not ReLU (e.g. binary), execute it here.
        if (!pd()->attr()->post_ops_.has_default_values()
                && !relu_fused_in_kernel) {
            CHECK(pd()->postops_.execute(ctx, dst, dst));
        }
    }
#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    else if (dt == data_type::f16) {
        const auto *src = static_cast<const dnnl::impl::float16_t *>(src_raw);
        auto *dst = static_cast<dnnl::impl::float16_t *>(dst_raw);

        if (is_max_pool) {
            MaxPooling_f16(src, dst, MB, C, OD, OH, OW, ID, IH, IW, KD, KH, KW,
                    SD, SH, SW, padF, padT, padL);
        } else if (is_avg_pool_include) {
            AvgPoolingIncludePadding_f16(src, dst, MB, C, OD, OH, OW, ID, IH,
                    IW, KD, KH, KW, SD, SH, SW, padF, padT, padL);
        } else if (is_avg_pool_exclude) {
            AvgPoolingExcludePadding_f16(src, dst, MB, C, OD, OH, OW, ID, IH,
                    IW, KD, KH, KW, SD, SH, SW, padF, padT, padL);
        } else {
            return status::unimplemented;
        }
    }
#endif
    else {
        return status::unimplemented;
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
