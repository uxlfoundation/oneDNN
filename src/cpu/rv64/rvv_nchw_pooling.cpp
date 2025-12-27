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
#include <riscv_vector.h>

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/stream.hpp"
#include "cpu/rv64/rvv_nchw_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {
void MaxPooling(const float *src, float *dst, const dim_t batch,
        const dim_t channels, const dim_t outD, const dim_t outH,
        const dim_t outW, const dim_t inD, const dim_t inH, const dim_t inW,
        const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft) {

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
        int iw_end = std::min(ow_offset + kerW, inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = -__FLT_MAX__;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-__FLT_MAX__, cycleLength);

        for (int id = std::max(od_offset, 0);
                id < std::min(od_offset + kerD, inD); id++)
            for (int ih = std::max(oh_offset, 0);
                    ih < std::min(oh_offset + kerH, inH); ih++) {
                const auto local_src_offset
                        = IWH * id + (size_t)inW * ih + std::max(ow_offset, 0);

                size_t iw = 0;
                for (; iw < size - cycleLength; iw += cycleLength) {
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

        vfloat32m1_t min_scalar = __riscv_vfmv_v_f_f32m1(-__FLT_MAX__, 1);

        cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vred_res;
        vred_res = __riscv_vfredmax_vs_f32m1_f32m1(
                vmax, min_scalar, cycleLength);

        __riscv_vse32_v_f32m1(&dst[dst_offset], vred_res, 1);
    });
}

void AvgPoolingIncludePadding(const float *src, float *dst, const dim_t batch,
        const dim_t channels, const dim_t outD, const dim_t outH,
        const dim_t outW, const dim_t inD, const dim_t inH, const dim_t inW,
        const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft) {

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
        int iw_end = std::min(ow_offset + kerW, inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = 0.0f;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, cycleLength);

        for (int id = std::max(od_offset, 0);
                id < std::min(od_offset + kerD, inD); id++)
            for (int ih = std::max(oh_offset, 0);
                    ih < std::min(oh_offset + kerH, inH); ih++) {
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

        float zero = 0.0f;
        vfloat32m1_t zero_scalar = __riscv_vfmv_v_f_f32m1(zero, 1);

        cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vred_res;
        vred_res = __riscv_vfredusum_vs_f32m1_f32m1(
                vsum, zero_scalar, cycleLength);

        float red_res;
        __riscv_vse32_v_f32m1(&red_res, vred_res, 1);
        dst[dst_offset] = red_res / kernel_volume;
    });
}

void AvgPoolingExcludePadding(const float *src, float *dst, const dim_t batch,
        const dim_t channels, const dim_t outD, const dim_t outH,
        const dim_t outW, const dim_t inD, const dim_t inH, const dim_t inW,
        const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft) {

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
        int iw_end = std::min(ow_offset + kerW, inW);

        if (iw_start >= iw_end) {
            dst[dst_offset] = 0.0f;
            return;
        }

        size_t size = iw_end - iw_start;
        size_t cycleLength = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, cycleLength);

        size_t count = 0;

        for (int id = od_offset; id < od_offset + kerD; id++) {
            if (id < 0 || id >= inD) continue;
            for (int ih = oh_offset; ih < oh_offset + kerH; ih++) {
                if (ih < 0 || ih >= inH) continue;

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

#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
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

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto dt = src_d.data_type();

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
        auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC) + src_d.off_l(0);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST) + dst_d.off_l(0);

        if (is_max_pool) {
            MaxPooling(src, dst, MB, C, OD, OH, OW, ID, IH, IW, KD, KH, KW, SD,
                    SH, SW, padF, padT, padL);
        } else if (is_avg_pool_exclude) {
            AvgPoolingExcludePadding(src, dst, MB, C, OD, OH, OW, ID, IH, IW,
                    KD, KH, KW, SD, SH, SW, padF, padT, padL);
        } else if (is_avg_pool_include) {
            AvgPoolingIncludePadding(src, dst, MB, C, OD, OH, OW, ID, IH, IW,
                    KD, KH, KW, SD, SH, SW, padF, padT, padL);
        } else {
            return status::unimplemented;
        }

        if (!pd()->attr()->post_ops_.has_default_values()) {
            CHECK(pd()->postops_.execute(ctx, dst, dst));
        }
    }
#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    else if (dt == data_type::f16) {
        auto src = CTX_IN_MEM(const dnnl::impl::float16_t *, DNNL_ARG_SRC)
                + src_d.off_l(0);
        auto dst = CTX_OUT_MEM(dnnl::impl::float16_t *, DNNL_ARG_DST)
                + dst_d.off_l(0);

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
