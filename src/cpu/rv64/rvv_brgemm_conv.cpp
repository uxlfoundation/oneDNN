/*******************************************************************************
* Copyright 2026 ZTE Corporation
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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/rv64/rvv_brgemm_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;

status_t rvv_brgemm_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;

    VDISPATCH_CONV(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    const auto dst_type = dst_md(0)->data_type;
    VDISPATCH_CONV(attr()->has_default_values(
                           primitive_attr_t::skip_mask_t::post_ops, dst_type),
            VERBOSE_UNSUPPORTED_ATTR);

    // Only no-post-ops or sum-at-position-0 supported.
    const auto &po = attr()->post_ops_;
    bool post_ops_ok = true;
    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry_[i];
        if (e.is_sum()) {
            if (i != 0) {
                post_ops_ok = false;
                break;
            }
        } else {
            post_ops_ok = false;
            break;
        }
    }
    VDISPATCH_CONV(post_ops_ok, VERBOSE_UNSUPPORTED_POSTOP);

    VDISPATCH_CONV_SC(brgemm_convolution_utils::init_conf(jcp_, *desc(),
                              src_md_, weights_md_, dst_md_, bias_md_, attr_,
                              dnnl_get_max_threads()),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "brgemm_conv");

    return status::success;
}

status_t rvv_brgemm_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto &jcp = pd()->jcp_;

    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto wei = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto bia = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const int G = jcp.ngroups;
    const int IC = jcp.ic; // per group
    const int OC = jcp.oc; // per group
    const int IC_all = G * IC;
    const int OC_all = G * OC;

    const int IH = (jcp.ndims >= 4) ? jcp.ih : 1;
    const int IW = jcp.iw;
    const int OD = (jcp.ndims >= 5) ? jcp.od : 1;
    const int OH = (jcp.ndims >= 4) ? jcp.oh : 1;
    const int OW = jcp.ow;

    const int KD = (jcp.ndims >= 5) ? jcp.kd : 1;
    const int KH = (jcp.ndims >= 4) ? jcp.kh : 1;
    const int KW = jcp.kw;

    const int SD = (jcp.ndims >= 5) ? jcp.stride_d : 1;
    const int SH = (jcp.ndims >= 4) ? jcp.stride_h : 1;
    const int SW = jcp.stride_w;

    // Dilation step = dilate + 1.
    const int DD = (jcp.ndims >= 5) ? jcp.dilate_d + 1 : 1;
    const int DH = (jcp.ndims >= 4) ? jcp.dilate_h + 1 : 1;
    const int DW = jcp.dilate_w + 1;

    const int FP = (jcp.ndims >= 5) ? jcp.f_pad : 0;
    const int TP = (jcp.ndims >= 4) ? jcp.t_pad : 0;
    const int LP = jcp.l_pad;

    const int ID = (jcp.ndims >= 5) ? jcp.id : 1;

    // Source strides (nhwc / nwc / ndhwc).
    const dim_t src_w_str = IC_all;
    const dim_t src_h_str = static_cast<dim_t>(IW) * IC_all;
    const dim_t src_d_str = static_cast<dim_t>(IH) * src_h_str;
    const dim_t src_mb_str = static_cast<dim_t>(ID) * src_d_str;

    // Dest strides.
    const dim_t dst_h_str = static_cast<dim_t>(OW) * OC_all;
    const dim_t dst_d_str = static_cast<dim_t>(OH) * dst_h_str;
    const dim_t dst_mb_str = static_cast<dim_t>(OD) * dst_d_str;

    // Weight strides (hwio / dhwio / hwigo / dhwigo).
    const dim_t wei_kpos_str = static_cast<dim_t>(IC) * OC_all;

    // GEMM leading dimensions (column-major / BLAS convention).
    const dim_t LDA = OC_all; // weight: M=OC, K=IC
    const dim_t LDB = static_cast<dim_t>(SW) * IC_all; // source
    const dim_t LDC = OC_all; // destination

    const float one = 1.0f;

    // When SH=SW=1 and IW=OW (padding-preserved spatial), consecutive
    // output rows correspond to contiguous source/dst in nhwc layout.
    const bool can_batch_rows = (SH == 1 && SW == 1 && IW == OW);

    if (can_batch_rows) {
        // Parallel over (mb, groups, od) — each thread handles all OH rows.
        parallel_nd(jcp.mb, G, OD, [&](dim_t n, dim_t g, dim_t od) {
            float *dst_base = dst + n * dst_mb_str + od * dst_d_str + g * OC;

            // Zero entire output slice for this (n, g, od).
            if (!jcp.with_sum) {
                for (dim_t s = 0; s < OH * OW; s++) {
                    float *d = dst_base + s * OC_all;
                    for (dim_t oc = 0; oc < OC; oc++)
                        d[oc] = 0.0f;
                }
            }

            for (int kd = 0; kd < KD; kd++) {
                const int id = od * SD + kd * DD - FP;
                if (id < 0 || id >= ID) continue;

                for (int kh = 0; kh < KH; kh++) {
                    // Valid oh range: need 0 <= oh + kh*DH - TP < IH.
                    const int oh_s
                            = nstl::max(0, TP - kh * DH); // DH already +1
                    const int oh_e = nstl::min(OH, IH + TP - kh * DH);
                    if (oh_s >= oh_e) continue;
                    const int ih_start = oh_s + kh * DH - TP; // SH=1

                    for (int kw = 0; kw < KW; kw++) {
                        const int iw_base = kw * DW - LP;
                        int ow_s = iw_base < 0 ? -iw_base : 0;
                        int ow_e = nstl::min(OW, IW - iw_base);
                        const dim_t valid_ow = ow_e - ow_s;
                        if (valid_ow <= 0) continue;
                        const int iw_start = iw_base + ow_s; // SW=1

                        const float *A = wei
                                + ((kd * KH + kh) * KW + kw) * wei_kpos_str
                                + g * OC;

                        if (ow_s == 0 && ow_e == OW) {
                            // Full-width: source/dst contiguous across rows.
                            dim_t num_rows = oh_e - oh_s;
                            dim_t N = num_rows * OW;
                            const float *B = src + n * src_mb_str
                                    + id * src_d_str + ih_start * src_h_str
                                    + iw_start * src_w_str + g * IC;
                            float *C = dst_base + oh_s * dst_h_str
                                    + ow_s * OC_all;
                            dim_t M = OC, K_dim = IC;
                            status_t st = extended_sgemm("N", "N", &M, &N,
                                    &K_dim, &one, A, &LDA, B, &LDB, &one, C,
                                    &LDC);
                            if (st != status::success) return;
                        } else {
                            // Edge (partial width): per-row GEMM.
                            for (int oh = oh_s; oh < oh_e; oh++) {
                                const int ih = oh + kh * DH - TP;
                                const float *B = src + n * src_mb_str
                                        + id * src_d_str + ih * src_h_str
                                        + iw_start * src_w_str + g * IC;
                                float *C = dst_base + oh * dst_h_str
                                        + ow_s * OC_all;
                                dim_t M = OC, N_dim = valid_ow, K_dim = IC;
                                status_t st = extended_sgemm("N", "N", &M,
                                        &N_dim, &K_dim, &one, A, &LDA, B, &LDB,
                                        &one, C, &LDC);
                                if (st != status::success) return;
                            }
                        }
                    }
                }
            }

            if (jcp.with_bias) {
                const float *bia_g = bia + g * OC;
                for (dim_t s = 0; s < OH * OW; s++) {
                    float *d = dst_base + s * OC_all;
                    for (dim_t oc = 0; oc < OC; oc++)
                        d[oc] += bia_g[oc];
                }
            }
        });
    } else {
        // Strided convolutions: parallel over (mb, groups, od, oh).
        const dim_t work = static_cast<dim_t>(jcp.mb) * G * OD * OH;

        parallel(jcp.nthr, [&](const int ithr, const int nthr) {
            dim_t start {0}, end {0};
            balance211(work, nthr, ithr, start, end);

            int n {0}, g {0}, od {0}, oh {0};
            nd_iterator_init(start, n, jcp.mb, g, G, od, OD, oh, OH);

            for (dim_t iwork = start; iwork < end; iwork++) {
                float *dst_row = dst + n * dst_mb_str + od * dst_d_str
                        + oh * dst_h_str + g * OC;

                if (!jcp.with_sum) {
                    for (int ow = 0; ow < OW; ow++)
                        for (int oc = 0; oc < OC; oc++)
                            dst_row[ow * OC_all + oc] = 0.0f;
                }

                for (int kd = 0; kd < KD; kd++) {
                    const int id = od * SD + kd * DD - FP;
                    if (id < 0 || id >= ID) continue;

                    for (int kh = 0; kh < KH; kh++) {
                        const int ih = oh * SH + kh * DH - TP;
                        if (ih < 0 || ih >= IH) continue;

                        for (int kw = 0; kw < KW; kw++) {
                            const int iw_base = kw * DW - LP;
                            int ow_s = 0;
                            if (iw_base < 0) ow_s = (-iw_base + SW - 1) / SW;
                            int ow_e = nstl::min(
                                    OW, (IW - iw_base + SW - 1) / SW);
                            const dim_t valid_ow = ow_e - ow_s;
                            if (valid_ow <= 0) continue;
                            const int iw_start = iw_base + ow_s * SW;

                            const float *A = wei
                                    + ((kd * KH + kh) * KW + kw) * wei_kpos_str
                                    + g * OC;
                            const float *B = src + n * src_mb_str
                                    + id * src_d_str + ih * src_h_str
                                    + iw_start * src_w_str + g * IC;
                            float *C = dst_row + ow_s * OC_all;

                            dim_t M = OC, N = valid_ow, K_dim = IC;
                            status_t st = extended_sgemm("N", "N", &M, &N,
                                    &K_dim, &one, A, &LDA, B, &LDB, &one, C,
                                    &LDC);
                            if (st != status::success) return;
                        }
                    }
                }

                if (jcp.with_bias) {
                    const float *bia_g = bia + g * OC;
                    for (int ow = 0; ow < OW; ow++) {
                        float *d = dst_row + ow * OC_all;
                        for (int oc = 0; oc < OC; oc++)
                            d[oc] += bia_g[oc];
                    }
                }

                nd_iterator_step(n, jcp.mb, g, G, od, OD, oh, OH);
            }
        });
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
