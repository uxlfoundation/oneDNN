/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/x64/jit_avx512_core_amx_conv_utils.hpp"
#include "cpu/x64/jit_avx512_core_amx_deconvolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

// NOTE: This primitive has identical kernel to bwd/d convolution. Hence, all
//       parameters stored in `pd()->jcp_` are in terms of bwd/d convolution.
//       This means that the following parameters have been exchanged:
//         1. ic <-> oc
//         2. ih <-> oh
//         3. iw <-> ow
//       The same exchange applies to all derivative values in `pd()->jcp_`
//       (eg, ic_block <-> oc_block, etc).

status_t jit_avx512_core_amx_deconvolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_ic % jcp.nb_ic_blocking == 0);

    const size_t src_dt_size = jcp.typesize_in;
    const size_t wei_dt_size = jcp.typesize_in;
    const size_t bia_dt_size = jcp.typesize_bia;
    const size_t dst_dt_size = jcp.typesize_out;

    const dim_t wei_g_shift = wht_blk_off(weights_d, 1, 0);
    const dim_t wei_ic_shift = wht_blk_off(weights_d, 0, jcp.nb_ic_blocking);
    const size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

    auto inp_p_buffer = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_inp_buffer);
    auto wsp = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_amx_wsp_buffer);
    auto global_tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);

    const int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    const int ih_chunks = utils::div_up(jcp.ih, jcp.ih_blk_size);
    const int work_amount
            = jcp.mb * jcp.ngroups * jcp.id * ih_chunks * jcp.nb_iw * ic_chunks;

    const bool is_1d = jcp.ndims == 3;
    const bool is_3d = jcp.ndims == 5;

    auto *padded_bias = ctx.get_scratchpad_grantor().template get<char>(
            memory_tracking::names::key_conv_padded_bias);
    const bool use_padded_bias
            = jcp.with_bias && jcp.ic != jcp.ic_without_padding;
    auto *bias_ptr = use_padded_bias ? padded_bias : bias;

    parallel(1, [=](const int ithr, const int nthr) {
        if (jcp.with_bias && jcp.ic != jcp.ic_without_padding) {
            utils::array_copy(
                    padded_bias, bias, bia_dt_size * jcp.ic_without_padding);
            utils::array_set(padded_bias + bia_dt_size * jcp.ic_without_padding,
                    0.f, bia_dt_size * (jcp.ic - jcp.ic_without_padding));
        }
    });

    parallel(jcp.nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_args_t();

        char *const __restrict tcfg = global_tcfg + ithr * AMX_PALETTE_SIZE;
        kernel_->tile_configure(tcfg);
        amx_tile_configure(tcfg);

        amx_utils::spatial_features_3d_t sfd(jcp);

        float *dst_scales_inv_ptr = nullptr;
        if (jcp.with_dst_scales) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr
                    = ctx.get_scratchpad_grantor().template get<float>(
                              key_conv_dst_scales)
                    + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        int mb {0}, g {0}, id_s {0}, ihc {0}, iwb {0}, icc {0};
        nd_iterator_init(start, mb, jcp.mb, g, jcp.ngroups, id_s, jcp.id, ihc,
                ih_chunks, iwb, jcp.nb_iw, icc, ic_chunks);
        int last_copied_mb = -1;
        int last_copied_id = -1;
        int last_copied_ihc = -1;
        int last_copied_iwb = -1;
        int last_copied_g = -1;
        while (start < end) {
            char *inp_buffer
                    = inp_p_buffer + ithr * jcp.inp_buffer_size * src_dt_size;

            assert(IMPLICATION(
                    jcp.ngroups > 1, jcp.ic == jcp.ic_without_padding));
            int ic = g * jcp.ic + icc * jcp.nb_ic_blocking * jcp.ic_block;
            int icb = jcp.is_nspc ? ic : ic / jcp.ic_block;
            assert(IMPLICATION(
                    jcp.ngroups > 1, jcp.oc == jcp.oc_without_padding));
            const int ocb = g * (jcp.is_nspc ? jcp.oc : jcp.nb_oc);
            auto bias_w = bias_ptr
                    ? bias_ptr + (bias_d.blk_off(ic) * bia_dt_size)
                    : nullptr;

            const int ih_b = ihc * jcp.ih_blk_size;
            const int ih_e = nstl::min(jcp.ih, ih_b + jcp.ih_blk_size);
            const int iw = iwb * jcp.iw_block;
            bool is_inp_buffer_relevant = true && last_copied_mb == mb
                    && last_copied_id == id_s && last_copied_ihc == ihc
                    && last_copied_iwb == iwb && last_copied_g == g;

            sfd.update_params(id_s);
            p.kd_padding = sfd.get_filter_padding();
            const int d_lo = sfd.get_lower_offset();
            const int d_oj = sfd.get_output_offset();

            int ih_step = jcp.nb_ih_blocking;
            for (int ih = ih_b; ih < ih_e; ih += ih_step) {
                if (!is_inp_buffer_relevant) {
                    const int gen_kh = (jcp.kh - 1) * (jcp.dilate_h + 1) + 1;
                    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
                    // dox: x-index dilated by strides (dox = ox * stride_x)
                    const int doh = ih + jcp.t_pad - (gen_kh - 1);
                    const int dow = iw + jcp.l_pad - (gen_kw - 1);
                    const int doh_b = ih_b + jcp.t_pad - (gen_kh - 1);
                    const int doh_l = (jcp.oh - 1) * jcp.stride_h; // last oh
                    const int dow_l = (jcp.ow - 1) * jcp.stride_w; // last ow

                    // dox_{s,f}: start and finish indices for copy kernel
                    const int doh_s = doh + (ih == ih_b ? 0 : gen_kh - 1);
                    const int doh_f = doh + (ih_step - 1) + (gen_kh - 1);
                    const int delta_h = doh_f - doh_s + 1;
                    const int doh_t_overflow = 0 < doh_s && doh_s < doh_l
                            ? nstl::additive_inverse_modulo(doh_s, jcp.stride_h)
                            : nstl::max(0, -doh_s);
                    const int doh_b_overflow = 0 < doh_f && doh_f < doh_l
                            ? nstl::modulo(doh_f, jcp.stride_h)
                            : nstl::max(0, nstl::min(delta_h, doh_f - doh_l));
                    int dow_s = dow;
                    int dow_f = dow + jcp.owp - 1;
                    const int delta_w = dow_f - dow_s + 1;
                    const int dow_l_overflow = 0 < dow_s && dow_s < dow_l
                            ? nstl::additive_inverse_modulo(dow_s, jcp.stride_w)
                            : nstl::max(0, -dow_s);
                    const int dow_r_overflow = 0 < dow_f && dow_f < dow_l
                            ? nstl::modulo(dow_f, jcp.stride_w)
                            : nstl::max(0, nstl::min(delta_w, dow_f - dow_l));
                    const int oh_s
                            = nstl::max(0, utils::div_up(doh_s, jcp.stride_h));
                    const int ow_s
                            = nstl::max(0, utils::div_up(dow_s, jcp.stride_w));
                    // how many real data rows to copy (including padding)
                    p.t_overflow = nstl::min(delta_h, doh_t_overflow);
                    p.b_overflow = nstl::min<size_t>(
                            delta_h - p.t_overflow, doh_b_overflow);
                    p.kh_padding = nstl::max<size_t>(
                            0, delta_h - p.t_overflow - p.b_overflow);
                    p.l_overflow = nstl::min(delta_w, dow_l_overflow);
                    p.kw_padding = nstl::max<size_t>(
                            0, delta_w - dow_l_overflow - dow_r_overflow);
                    p.r_overflow = nstl::min<size_t>(
                            delta_w - dow_l_overflow, dow_r_overflow);
                    size_t inp_offset = is_1d ? src_d.blk_off(mb, ocb, ow_s)
                            : is_3d ? src_d.blk_off(mb, ocb, d_oj, oh_s, ow_s)
                                    : src_d.blk_off(mb, ocb, oh_s, ow_s);
                    p.src = src + src_dt_size * inp_offset;
                    p.dst = inp_buffer
                            + (size_t)(doh_s - doh_b) * jcp.owp
                                    * jcp.oc_block_int * src_dt_size;

                    kernel_->bwd_data_copy_kernel()(&p);
                }

                size_t dst_offset = is_1d ? dst_d.blk_off(mb, icb, iw)
                        : is_3d           ? dst_d.blk_off(mb, icb, id_s, ih, iw)
                                          : dst_d.blk_off(mb, icb, ih, iw);
                p.dst = inp_buffer
                        + (size_t)(ih - ih_b) * jcp.owp * jcp.oc_block_int
                                * src_dt_size;
                p.src = dst + dst_dt_size * dst_offset;
                p.filt = weights
                        + wei_dt_size
                                * (g * wei_g_shift + icc * wei_ic_shift
                                        + d_lo * wht_d_stride);
                p.bias = bias_w;
                p.src_scales = src_scales;
                p.wei_scales = jcp.with_wei_scales
                        ? static_cast<const float *>(wei_scales)
                                + jcp.is_ic_scale * ic
                        : nullptr;
                p.dst_scales = dst_scales_inv_ptr;
                p.acc_s32 = wsp + ithr * jcp.wsp_buffer_size;
                p.last_h = (ih + ih_step <= ih_e);
                p.iwb = iwb;
                p.ic_blocks = icc * jcp.nb_ic_blocking;

                (*kernel_)(&p);
            }
            last_copied_mb = mb;
            last_copied_id = id_s;
            last_copied_ihc = ihc;
            last_copied_iwb = iwb;
            last_copied_g = g;
            ++start;
            nd_iterator_step(mb, jcp.mb, g, jcp.ngroups, id_s, jcp.id, ihc,
                    ih_chunks, iwb, jcp.nb_iw, icc, ic_chunks);
        }
        amx_tile_release();
    });
    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
