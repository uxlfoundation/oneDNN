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

#include "cpu/rv64/brgemm/brgemm.hpp"
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

    // Create the JIT BRGEMM kernel for the convolution's GEMM shape.
    const dim_t M = jcp_.oc;
    const dim_t K = jcp_.ic;
    const dim_t OC_all = static_cast<dim_t>(jcp_.ngroups) * jcp_.oc;
    const dim_t IC_all = static_cast<dim_t>(jcp_.ngroups) * jcp_.ic;
    const dim_t LDA = OC_all;
    const dim_t LDB = static_cast<dim_t>(jcp_.stride_w) * IC_all;
    const dim_t LDC = OC_all;

    brgemm_desc_t brg_desc;
    CHECK(brgemm_desc_init(&brg_desc, v, brgemm_strd, data_type::f32,
            data_type::f32, brgemm_col_major, 1.0f, 1.0f, LDA, LDB, LDC, M,
            jcp_.ow, K));

    brgemm_kernel_t *kernel = nullptr;
    CHECK(brgemm_kernel_create(&kernel, brg_desc));
    brg_kernel_.reset(kernel);

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

    const auto *brg_kernel = pd()->brg_kernel_.get();

    {
        // Parallel over (mb, groups, od, oh) for good multi-core scaling.
        // Each thread handles one or more output rows independently.
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

                bool first_kpos = !jcp.with_sum;

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

                            const float beta_val = first_kpos ? 0.0f : 1.0f;
                            brgemm_kernel_execute(
                                    brg_kernel, A, B, C, valid_ow, beta_val);
                            first_kpos = false;
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
