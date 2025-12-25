/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#include "cpu/rv64/jit_rvv_1x1_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

void jit_rvv_1x1_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    parallel(pd()->jcp_.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, dst, scratchpad);
    });
}

void jit_rvv_1x1_convolution_fwd_t::execute_forward_thr(const int ithr,
        const int nthr, const float *src, const float *weights,
        const float *bias, float *dst,
        const memory_tracking::grantor_t &scratchpad) const {

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    // RVV 1x1 convolution uses NHWC layout.
    // Spatial dimensions are collapsed into 'os'.
    // Threading is balanced over (MB * groups * nb_bcast) and (nb_load).

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
    int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};

    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
            ocb_start, ocb_end, jcp.load_grp_count);

    if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;

    auto p = jit_1x1_conv_args_t();

    auto ker_1x1 = [&](int ocb, int load_step, int icb, int n, int g, int osb,
                           int bcast_step) {
        const int oc_off = g * jcp.oc_without_padding + ocb * jcp.oc_block;
        const size_t dst_off
                = (size_t)n * jcp.os * jcp.ngroups * jcp.oc_without_padding
                + (size_t)osb * jcp.bcast_block * jcp.ngroups
                        * jcp.oc_without_padding
                + oc_off;

        p.output_data = &dst[dst_off];
        p.bias_data = bias ? &bias[oc_off] : nullptr;

        const size_t wei_off = (size_t)g * jcp.oc * jcp.ic_without_padding
                + (size_t)ocb * jcp.ic_without_padding * jcp.oc_block
                + (size_t)icb * jcp.ic_block * jcp.oc_block;
        p.load_data = &weights[wei_off];

        const int ic_off = g * jcp.ic_without_padding + icb * jcp.ic_block;
        const size_t src_off
                = (size_t)n * jcp.is * jcp.ngroups * jcp.ic_without_padding
                + (size_t)osb * jcp.bcast_block * jcp.ngroups
                        * jcp.ic_without_padding
                + ic_off;
        p.bcast_data = &src[src_off];

        p.bcast_dim = this_block_size(
                osb * jcp.bcast_block, jcp.os, bcast_step * jcp.bcast_block);
        p.load_dim = this_block_size(ocb * jcp.oc_block, jcp.oc_without_padding,
                load_step * jcp.oc_block);
        p.reduce_dim = this_block_size(icb * jcp.ic_block,
                jcp.ic_without_padding, jcp.nb_reduce_blocking * jcp.ic_block);

        p.first_last_flag = (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (icb + jcp.nb_reduce_blocking >= jcp.nb_reduce
                                ? FLAG_REDUCE_LAST
                                : 0);

        (*kernel_)(&p);
    };

    // Loop order: Load -> Bcast -> Reduce (LBR)
    // This order keeps weights in registers/L1 while iterating over spatial.
    for (int ocb = ocb_start; ocb < ocb_end;) {
        int load_step = step(
                jcp.nb_load_blocking, ocb_end - ocb, jcp.nb_load_blocking_max);
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n {0}, g {0}, osb {0};
            nd_iterator_init(
                    iwork, n, jcp.mb, g, jcp.ngroups, osb, jcp.nb_bcast);

            int bcast_step = step(jcp.nb_bcast_blocking, bcast_end - iwork,
                    jcp.nb_bcast_blocking_max);
            bcast_step = nstl::min(bcast_step, jcp.nb_bcast - osb);

            for (int icb = 0; icb < jcp.nb_reduce;
                    icb += jcp.nb_reduce_blocking) {
                ker_1x1(ocb, load_step, icb, n, g, osb, bcast_step);
            }
            iwork += bcast_step;
        }
        ocb += load_step;
    }
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
