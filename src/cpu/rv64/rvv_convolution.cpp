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

#include <algorithm>
#include <vector>
#include <riscv_vector.h>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/rv64/rvv_convolution.hpp"
#include "cpu/rv64/rvv_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t rvv_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {

    const data_type_t sdt = pd()->src_dt_;
    const data_type_t wdt = pd()->wei_dt_;
    const data_type_t ddt = pd()->dst_dt_;

    const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    const void *bias = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;
    void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const memory_desc_wrapper src_md(pd()->src_md());
    const memory_desc_wrapper dst_md(pd()->dst_md());
    const memory_desc_wrapper wei_md(pd()->weights_md());
    const memory_desc_wrapper bias_md(pd()->desc()->bias_desc);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    rvv_postops_t postops_handler(post_ops);

    const bool src_is_nhwc = pd()->src_is_nhwc_;
    const bool dst_is_nhwc = pd()->dst_is_nhwc_;

    const dim_t MB = pd()->MB_;
    const dim_t IC = pd()->IC_;
    const dim_t OC = pd()->OC_;
    const dim_t IH = pd()->IH_;
    const dim_t IW = pd()->IW_;
    const dim_t OH = pd()->OH_;
    const dim_t OW = pd()->OW_;
    const dim_t KH = pd()->KH_;
    const dim_t KW = pd()->KW_;
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t PH = pd()->padT();
    const dim_t PW = pd()->padL();

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);
    const int wei_scale_mask = pd()->attr()->scales_.get_mask(DNNL_ARG_WEIGHTS);

    auto &mem = ctx.get_scratchpad_grantor();
    void *wei_pack_void = mem.template get<void>(
            memory_tracking::names::key_conv_permuted_weights);
    void *src_nhwc_buf_void = src_is_nhwc
            ? nullptr
            : mem.template get<void>(memory_tracking::names::key_conv_tr_src);
    void *dst_nhwc_buf_void = dst_is_nhwc
            ? nullptr
            : mem.template get<void>(memory_tracking::names::key_conv_ncsp_dst);

    const void *wei_oihw_base
            = ptr_add_elems(wei, wdt, static_cast<size_t>(wei_md.off_l(0)));
    const void *src_base_in
            = ptr_add_elems(src, sdt, static_cast<size_t>(src_md.off_l(0)));

    {
        pack_weights_dispatch(
                wdt, wei_oihw_base, wei_pack_void, OC, IC, KH, KW);
    }

    const void *src_nhwc = nullptr;
    if (src_is_nhwc) {
        src_nhwc = src_base_in;
    } else {
        reorder_src_to_nhwc_dispatch(
                sdt, src_base_in, src_nhwc_buf_void, MB, IC, IH, IW);
        src_nhwc = src_nhwc_buf_void;
    }

    void *dst_nhwc_void = dst_is_nhwc ? (ptr_add_elems_mut(dst, ddt,
                                  static_cast<size_t>(dst_md.off_l(0))))
                                      : dst_nhwc_buf_void;

    auto dst_off_nhwc = [&](dim_t n, dim_t h, dim_t w, dim_t oc) {
        return (((n * OH + h) * OW + w) * OC + oc);
    };

    auto load_bias = [&](dim_t oc) -> float {
        if (!bias) return 0.f;
        const dim_t b_off = bias_md.off(oc);
        return io::load_float_value(bias_md.data_type(), bias, b_off);
    };

    parallel_nd(MB, OH, OW, [&](dim_t n, dim_t oh, dim_t ow) {
        const size_t n_base_off
                = (((static_cast<size_t>(n) * IH) + 0) * IW + 0) * IC;
        const void *src_base_hw = ptr_add_elems(src_nhwc, sdt, n_base_off);

        for (dim_t oc = 0; oc < OC; ++oc) {
            double acc_dot = 0.0;
            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh * SH - PH + kh;
                if (ih < 0 || ih >= IH) continue;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow * SW - PW + kw;
                    if (iw < 0 || iw >= IW) continue;
                    const size_t src_off_elems
                            = (static_cast<size_t>(ih) * IW + iw) * IC;
                    const void *sp
                            = ptr_add_elems(src_base_hw, sdt, src_off_elems);

                    const size_t wei_off_elems
                            = (((static_cast<size_t>(oc) * KH + kh) * KW + kw)
                                    * IC);
                    const void *wp
                            = ptr_add_elems(wei_pack_void, wdt, wei_off_elems);

                    float dot = compute_dot_ic_fwd(sdt, wdt, sp, wp, IC);
                    acc_dot += static_cast<double>(dot);
                }
            }
            const float out_scalar
                    = finalize_conv_acc(static_cast<float>(acc_dot),
                            load_bias(oc), src_scales, wei_scales, dst_scales,
                            wei_scale_mask, oc, postops_handler);

            const size_t doff = dst_off_nhwc(n, oh, ow, oc);
            if (ddt == data_type::f32) {
                reinterpret_cast<float *>(dst_nhwc_void)[doff] = out_scalar;
            } else if (ddt == data_type::f16) {
                reinterpret_cast<_Float16 *>(dst_nhwc_void)[doff]
                        = (_Float16)out_scalar;
            } else if (ddt == data_type::s32) {
                reinterpret_cast<int32_t *>(dst_nhwc_void)[doff]
                        = saturate_cast<int32_t>(out_scalar);
            } else if (ddt == data_type::s8) {
                reinterpret_cast<int8_t *>(dst_nhwc_void)[doff]
                        = saturate_cast<int8_t>(out_scalar);
            } else if (ddt == data_type::u8) {
                reinterpret_cast<uint8_t *>(dst_nhwc_void)[doff]
                        = saturate_cast<uint8_t>(out_scalar);
            }
        }
    });

    if (!dst_is_nhwc) {
        if (ddt == data_type::f32) {
            reorder_dst_nhwc_to_nchw<float>(
                    reinterpret_cast<const float *>(dst_nhwc_void),
                    reinterpret_cast<float *>(dst) + dst_md.off_l(0), MB, OC,
                    OH, OW);
        } else if (ddt == data_type::f16) {
            reorder_dst_nhwc_to_nchw<_Float16>(
                    reinterpret_cast<const _Float16 *>(dst_nhwc_void),
                    reinterpret_cast<_Float16 *>(dst) + dst_md.off_l(0), MB, OC,
                    OH, OW);
        } else if (ddt == data_type::s32) {
            reorder_dst_nhwc_to_nchw<int32_t>(
                    reinterpret_cast<const int32_t *>(dst_nhwc_void),
                    reinterpret_cast<int32_t *>(dst) + dst_md.off_l(0), MB, OC,
                    OH, OW);
        } else if (ddt == data_type::s8) {
            reorder_dst_nhwc_to_nchw<int8_t>(
                    reinterpret_cast<const int8_t *>(dst_nhwc_void),
                    reinterpret_cast<int8_t *>(dst) + dst_md.off_l(0), MB, OC,
                    OH, OW);
        } else if (ddt == data_type::u8) {
            reorder_dst_nhwc_to_nchw<uint8_t>(
                    reinterpret_cast<const uint8_t *>(dst_nhwc_void),
                    reinterpret_cast<uint8_t *>(dst) + dst_md.off_l(0), MB, OC,
                    OH, OW);
        }
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl