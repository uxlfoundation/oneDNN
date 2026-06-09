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
#include <float.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "cpu/rv64/jit_rvv_pooling_kernel.hpp"
#include "cpu/rv64/rvv_nhwc_pooling.hpp"

#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
#include <riscv_vector.h>
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

using kernel_t = jit_rvv_pooling_fwd_kernel_t;

// Compute per-output-pixel JIT kernel parameters (spatial window, init/scale).
kernel_t::call_params_t make_pooling_params(kernel_t::alg_t alg,
        const float *src_mb_base, float *dst, dim_t mb, dim_t channels,
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

    int id_start = std::max(od_offset, 0);
    int ih_start = std::max(oh_offset, 0);
    int iw_start = std::max(ow_offset, 0);
    int id_end = std::min(od_offset + (int)kerD, (int)inD);
    int ih_end = std::min(oh_offset + (int)kerH, (int)inH);
    int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

    float init_val, scale_val;
    if (alg == kernel_t::alg_t::max_pool) {
        init_val = -FLT_MAX;
        scale_val = 0.0f; /* unused for max pool */
    } else if (alg == kernel_t::alg_t::avg_include) {
        init_val = 0.0f;
        scale_val = 1.0f / (float)(kerD * kerH * kerW);
    } else { // avg_exclude
        init_val = 0.0f;
        const int count = std::max(0, id_end - id_start)
                * std::max(0, ih_end - ih_start)
                * std::max(0, iw_end - iw_start);
        scale_val = (count > 0) ? 1.0f / (float)count : 0.0f;
    }

    kernel_t::call_params_t p;
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
    p.w_spatial_byte_stride = channels * sizeof(float);
    p.init_val = init_val;
    p.scale_val = scale_val;
    p.relu_alpha = relu_alpha;
    p.with_relu = with_relu;
    p.src_vec_byte_stride = (dim_t)sizeof(float);
    p.dst_vec_byte_stride = (dim_t)sizeof(float);
    return p;
}

template <kernel_t::alg_t alg>
static void Pooling_f32_jit(const void *src_raw, void *dst_raw,
        const dim_t batch, const dim_t channels, const dim_t outD,
        const dim_t outH, const dim_t outW, const dim_t inD, const dim_t inH,
        const dim_t inW, const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft,
        const rvv_postops_t &postops_handler) {

    static const kernel_t kernel(alg);

    const float *src = static_cast<const float *>(src_raw);
    float *dst = static_cast<float *>(dst_raw);

    const bool with_relu = postops_handler.is_relu_postop();
    const float relu_alpha = postops_handler.relu_alpha();

    const dim_t inW_stride = inW * channels;
    const dim_t inD_stride = inH * inW * channels;

    parallel_nd(batch, outD, outH, outW,
            [&](dim_t mb, dim_t od, dim_t oh, dim_t ow) {
        const size_t mb_offset = (size_t)mb * (size_t)inD * inD_stride;
        const float *src_mb_base = &src[mb_offset];

        auto p = make_pooling_params(alg, src_mb_base, dst, mb, channels, od,
                oh, ow, outD, outH, outW, inD, inH, inW, kerD, kerH, kerW,
                strideD, strideH, strideW, padFront, padTop, padLeft,
                inW_stride, inD_stride, with_relu, relu_alpha);

        kernel(&p);
    });
}

#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
static void MaxPooling_f16(const void *src_raw, void *dst_raw,
        const dim_t batch, const dim_t channels, const dim_t outD,
        const dim_t outH, const dim_t outW, const dim_t inD, const dim_t inH,
        const dim_t inW, const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft,
        const rvv_postops_t &) {

    const dnnl::impl::float16_t *src
            = static_cast<const dnnl::impl::float16_t *>(src_raw);
    dnnl::impl::float16_t *dst = static_cast<dnnl::impl::float16_t *>(dst_raw);

    parallel_nd(batch, outD, outH, outW,
            [&](dim_t mb, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_spatial_base
                = ((((size_t)mb * outD + od) * outH + oh) * outW + ow)
                * channels;

        const int od_offset = (int)(od * strideD - padFront);
        const int oh_offset = (int)(oh * strideH - padTop);
        const int ow_offset = (int)(ow * strideW - padLeft);
        const int id_start = std::max(od_offset, 0);
        const int ih_start = std::max(oh_offset, 0);
        const int iw_start = std::max(ow_offset, 0);
        const int id_end = std::min(od_offset + (int)kerD, (int)inD);
        const int ih_end = std::min(oh_offset + (int)kerH, (int)inH);
        const int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

        const float f16_lowest
                = (float)nstl::numeric_limits<dnnl::impl::float16_t>::lowest();

        if (id_start >= id_end || ih_start >= ih_end || iw_start >= iw_end) {
            size_t oc = 0;
            while (oc < (size_t)channels) {
                size_t vl = __riscv_vsetvl_e16m1((size_t)channels - oc);
                vfloat16m1_t vfill
                        = __riscv_vfmv_v_f_f16m1((_Float16)f16_lowest, vl);
                __riscv_vse16_v_f16m1(
                        (_Float16 *)&dst[dst_spatial_base + oc], vfill, vl);
                oc += vl;
            }
            return;
        }

        size_t oc = 0;
        while (oc < (size_t)channels) {
            size_t vl = __riscv_vsetvl_e16m1((size_t)channels - oc);
            vfloat16m1_t vmax
                    = __riscv_vfmv_v_f_f16m1((_Float16)f16_lowest, vl);

            for (int id = id_start; id < id_end; ++id) {
                for (int ih = ih_start; ih < ih_end; ++ih) {
                    for (int iw = iw_start; iw < iw_end; ++iw) {
                        const size_t src_spatial_base
                                = ((((size_t)mb * inD + id) * inH + ih) * inW
                                          + iw)
                                * channels;
                        const dnnl::impl::float16_t *src_ptr
                                = &src[src_spatial_base + oc];
                        vfloat16m1_t vsrc = __riscv_vle16_v_f16m1(
                                (const _Float16 *)src_ptr, vl);
                        vmax = __riscv_vfmax_vv_f16m1(vmax, vsrc, vl);
                    }
                }
            }

            __riscv_vse16_v_f16m1(
                    (_Float16 *)&dst[dst_spatial_base + oc], vmax, vl);
            oc += vl;
        }
    });
}

static void AvgPoolingIncludePadding_f16(const void *src_raw, void *dst_raw,
        const dim_t batch, const dim_t channels, const dim_t outD,
        const dim_t outH, const dim_t outW, const dim_t inD, const dim_t inH,
        const dim_t inW, const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft,
        const rvv_postops_t &) {

    const dnnl::impl::float16_t *src
            = static_cast<const dnnl::impl::float16_t *>(src_raw);
    dnnl::impl::float16_t *dst = static_cast<dnnl::impl::float16_t *>(dst_raw);
    const float kernel_volume = (float)(kerD * kerH * kerW);

    parallel_nd(batch, outD, outH, outW,
            [&](dim_t mb, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_spatial_base
                = ((((size_t)mb * outD + od) * outH + oh) * outW + ow)
                * channels;

        const int od_offset = (int)(od * strideD - padFront);
        const int oh_offset = (int)(oh * strideH - padTop);
        const int ow_offset = (int)(ow * strideW - padLeft);
        const int id_start = std::max(od_offset, 0);
        const int ih_start = std::max(oh_offset, 0);
        const int iw_start = std::max(ow_offset, 0);
        const int id_end = std::min(od_offset + (int)kerD, (int)inD);
        const int ih_end = std::min(oh_offset + (int)kerH, (int)inH);
        const int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

        size_t oc = 0;
        while (oc < (size_t)channels) {
            size_t vl = __riscv_vsetvl_e16m1((size_t)channels - oc);
            vfloat32m2_t vsum_f32 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            if (id_start < id_end && ih_start < ih_end && iw_start < iw_end) {
                for (int id = id_start; id < id_end; ++id) {
                    for (int ih = ih_start; ih < ih_end; ++ih) {
                        for (int iw = iw_start; iw < iw_end; ++iw) {
                            const size_t src_spatial_base
                                    = ((((size_t)mb * inD + id) * inH + ih)
                                                      * inW
                                              + iw)
                                    * channels;
                            const dnnl::impl::float16_t *src_ptr
                                    = &src[src_spatial_base + oc];
                            vfloat16m1_t vsrc_f16 = __riscv_vle16_v_f16m1(
                                    (const _Float16 *)src_ptr, vl);
                            vfloat32m2_t vsrc_f32
                                    = __riscv_vfwcvt_f_f_v_f32m2(vsrc_f16, vl);
                            vsum_f32 = __riscv_vfadd_vv_f32m2(
                                    vsum_f32, vsrc_f32, vl);
                        }
                    }
                }
            }

            vfloat32m2_t vscale_f32
                    = __riscv_vfmv_v_f_f32m2(1.0f / kernel_volume, vl);
            vfloat32m2_t vout_f32
                    = __riscv_vfmul_vv_f32m2(vsum_f32, vscale_f32, vl);

            vfloat16m1_t vout_f16 = __riscv_vfncvt_f_f_w_f16m1(vout_f32, vl);

            __riscv_vse16_v_f16m1(
                    (_Float16 *)&dst[dst_spatial_base + oc], vout_f16, vl);
            oc += vl;
        }
    });
}

static void AvgPoolingExcludePadding_f16(const void *src_raw, void *dst_raw,
        const dim_t batch, const dim_t channels, const dim_t outD,
        const dim_t outH, const dim_t outW, const dim_t inD, const dim_t inH,
        const dim_t inW, const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft,
        const rvv_postops_t &) {

    const dnnl::impl::float16_t *src
            = static_cast<const dnnl::impl::float16_t *>(src_raw);
    dnnl::impl::float16_t *dst = static_cast<dnnl::impl::float16_t *>(dst_raw);

    parallel_nd(batch, outD, outH, outW,
            [&](dim_t mb, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_spatial_base
                = ((((size_t)mb * outD + od) * outH + oh) * outW + ow)
                * channels;

        const int od_offset = (int)(od * strideD - padFront);
        const int oh_offset = (int)(oh * strideH - padTop);
        const int ow_offset = (int)(ow * strideW - padLeft);

        size_t oc = 0;
        while (oc < (size_t)channels) {
            size_t vl = __riscv_vsetvl_e16m1((size_t)channels - oc);
            vfloat32m2_t vsum_f32 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            float count_scalar = 0.0f;

            for (int id = od_offset; id < od_offset + (int)kerD; ++id) {
                if (id < 0 || id >= (int)inD) continue;
                for (int ih = oh_offset; ih < oh_offset + (int)kerH; ++ih) {
                    if (ih < 0 || ih >= (int)inH) continue;

                    int iw_start = std::max(ow_offset, 0);
                    int iw_end = std::min(ow_offset + (int)kerW, (int)inW);
                    if (iw_start >= iw_end) continue;

                    for (int iw = iw_start; iw < iw_end; ++iw) {
                        const size_t src_spatial_base
                                = ((((size_t)mb * inD + id) * inH + ih) * inW
                                          + iw)
                                * channels;
                        const dnnl::impl::float16_t *src_ptr
                                = &src[src_spatial_base + oc];
                        vfloat16m1_t vsrc_f16 = __riscv_vle16_v_f16m1(
                                (const _Float16 *)src_ptr, vl);
                        vfloat32m2_t vsrc_f32
                                = __riscv_vfwcvt_f_f_v_f32m2(vsrc_f16, vl);
                        vsum_f32 = __riscv_vfadd_vv_f32m2(
                                vsum_f32, vsrc_f32, vl);
                        count_scalar += 1.0f;
                    }
                }
            }

            if (count_scalar == 0.0f) {
                vfloat32m2_t vzero_f32 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
                vfloat16m1_t vout_f16
                        = __riscv_vfncvt_f_f_w_f16m1(vzero_f32, vl);
                __riscv_vse16_v_f16m1(
                        (_Float16 *)&dst[dst_spatial_base + oc], vout_f16, vl);
            } else {
                vfloat32m2_t vscale_f32
                        = __riscv_vfmv_v_f_f32m2(1.0f / count_scalar, vl);
                vfloat32m2_t vout_f32
                        = __riscv_vfmul_vv_f32m2(vsum_f32, vscale_f32, vl);

                vfloat16m1_t vout_f16
                        = __riscv_vfncvt_f_f_w_f16m1(vout_f32, vl);
                __riscv_vse16_v_f16m1(
                        (_Float16 *)&dst[dst_spatial_base + oc], vout_f16, vl);
            }
            oc += vl;
        }
    });
}
#endif // DNNL_RISCV_USE_ZVFH_INTRINSICS

} // namespace

riscv_nhwc_pooling_fwd_t::riscv_nhwc_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

status_t riscv_nhwc_pooling_fwd_t::execute_forward(
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

    rvv_postops_t postops_handler(pd()->attr()->post_ops_);

    const auto alg = pd()->desc()->alg_kind;
    const bool is_max_pool = alg == alg_kind::pooling_max;
    const bool is_avg_pool_include
            = alg == alg_kind::pooling_avg_include_padding;
    const bool is_avg_pool_exclude
            = alg == alg_kind::pooling_avg_exclude_padding;

    if (dt == data_type::f32) {
        using kalg = kernel_t::alg_t;
        if (is_max_pool) {
            Pooling_f32_jit<kalg::max_pool>(src_raw, dst_raw, MB, C, OD, OH, OW,
                    ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT, padL,
                    postops_handler);
        } else if (is_avg_pool_exclude) {
            Pooling_f32_jit<kalg::avg_exclude>(src_raw, dst_raw, MB, C, OD, OH,
                    OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT, padL,
                    postops_handler);
        } else if (is_avg_pool_include) {
            Pooling_f32_jit<kalg::avg_include>(src_raw, dst_raw, MB, C, OD, OH,
                    OW, ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT, padL,
                    postops_handler);
        } else {
            return status::unimplemented;
        }
    }
#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    // TODO: f16 post-ops not yet supported — postops_handler is accepted but
    // ignored by the f16 intrinsics path (enforced by post_ops_ok check in pd).
    else if (dt == data_type::f16) {
        if (is_max_pool) {
            MaxPooling_f16(src_raw, dst_raw, MB, C, OD, OH, OW, ID, IH, IW, KD,
                    KH, KW, SD, SH, SW, padF, padT, padL, postops_handler);
        } else if (is_avg_pool_exclude) {
            AvgPoolingExcludePadding_f16(src_raw, dst_raw, MB, C, OD, OH, OW,
                    ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT, padL,
                    postops_handler);
        } else if (is_avg_pool_include) {
            AvgPoolingIncludePadding_f16(src_raw, dst_raw, MB, C, OD, OH, OW,
                    ID, IH, IW, KD, KH, KW, SD, SH, SW, padF, padT, padL,
                    postops_handler);
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
