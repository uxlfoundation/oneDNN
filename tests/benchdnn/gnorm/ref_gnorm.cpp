/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "utils/parallel.hpp"

#include "gnorm/gnorm.hpp"

namespace gnorm {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const auto src = args.find(DNNL_ARG_SRC).get_host_f32_handle();
    const auto mean = args.find(DNNL_ARG_MEAN).get_host_f32_handle();
    const auto var = args.find(DNNL_ARG_VARIANCE).get_host_f32_handle();
    const auto sc = args.find(DNNL_ARG_SCALE).get_host_f32_handle();
    const auto sh = args.find(DNNL_ARG_SHIFT).get_host_f32_handle();
    auto dst = args.find(DNNL_ARG_DST).get_host_f32_handle();
    const auto src_scale = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC)
                                   .get_host_f32_handle();
    const auto dst_scale = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
                                   .get_host_f32_handle();

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_src_scale, src_scale.nelems() == 1));
    assert(IMPLICATION(has_dst_scale, dst_scale.nelems() == 1));

    const float src_scale_val = has_src_scale ? src_scale[0] : 1.f;
    const float dst_scale_val = has_dst_scale ? dst_scale[0] : 1.f;
    const float r_dst_scale_val = 1.0f / dst_scale_val;

    const int64_t MB = prb->mb;
    const int64_t G = prb->g;
    const int64_t D = prb->id;
    const int64_t H = prb->ih;
    const int64_t W = prb->iw;
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    auto v_po_masks = prb->attr.post_ops.get_po_masks(prb->ndims);

    benchdnn_parallel_nd(MB, G, [&](int64_t n, int64_t g) {
        float smean = mean[n * G + g];
        float svar = var[n * G + g];
        float sqrt_var = sqrtf(svar + prb->eps);
        float rcp_denom = 1.f / sqrt_var;

        for (int64_t c = prb->get_c_start(g); c < prb->get_c_start(g + 1);
                ++c) {
            float gamma = use_sc ? sc[c] : 1.f;
            float beta = use_sh ? sh[c] : 0.f;

            for_(int64_t d = 0; d < D; ++d)
            for_(int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w) {
                auto off = data_off(prb, n, c, d, h, w);
                float x_hat = (src[off] - smean) * rcp_denom;
                float res = gamma * x_hat + beta;
                const auto v_po_vals
                        = prepare_po_vals(dst, args, v_po_masks, off);
                res *= src_scale_val;
                maybe_post_ops(prb->attr, res, 0.f, v_po_vals);
                dst[off] = res * r_dst_scale_val;
            }
        }
    });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    auto src = args.find(DNNL_ARG_SRC).get_host_f32_handle();
    const auto mean = args.find(DNNL_ARG_MEAN).get_host_f32_handle();
    const auto var = args.find(DNNL_ARG_VARIANCE).get_host_f32_handle();
    const auto d_dst = args.find(DNNL_ARG_DIFF_DST).get_host_f32_handle();
    const auto sc = args.find(DNNL_ARG_SCALE).get_host_f32_handle();
    auto d_src = args.find(DNNL_ARG_DIFF_SRC).get_host_f32_handle();
    auto d_sc = args.find(DNNL_ARG_DIFF_SCALE).get_host_f32_handle();
    auto d_sh = args.find(DNNL_ARG_DIFF_SHIFT).get_host_f32_handle();

    const int64_t MB = prb->mb;
    const int64_t G = prb->g;
    const int64_t C = prb->ic;
    const int64_t D = prb->id;
    const int64_t H = prb->ih;
    const int64_t W = prb->iw;
    const bool glob_stats = prb->flags & GLOB_STATS;
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    const int C_PER_G = C / G;
    const float CSP = C_PER_G * D * H * W;

    // Scale and shift are computed over a channel, thus, accumulate diff_dst
    // values over the spatial only.
    benchdnn_parallel_nd(C, [&](int64_t c) {
        int64_t g = c / C_PER_G;

        float d_gamma = 0;
        float d_beta = 0;

        for (int64_t mb = 0; mb < MB; ++mb) {
            int64_t stat_off = mb * G + g;
            float smean = mean[stat_off];
            float svar = var[stat_off];
            float rcp_denom = 1.f / sqrtf(svar + prb->eps);

            for_(int64_t d = 0; d < D; ++d)
            for_(int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w) {
                auto off = data_off(prb, mb, c, d, h, w);
                float dd = d_dst[off];
                float src_hat = (src[off] - smean) * rcp_denom;
                d_gamma += dd * src_hat;
                d_beta += dd;
                // Update `src` to save on some computations.
                src[off] = src_hat;
            }
        }

        if (use_sc && (prb->dir & FLAG_WEI)) d_sc[c] = d_gamma;
        if (use_sh && (prb->dir & FLAG_WEI)) d_sh[c] = d_beta;
    });

    // Statistics values are computed over the `group * spatial`, it's the unit
    // where a single source point has its impact and derivative values
    // distributed, thus, parallel over MB and G - independent clusters of a
    // tensor.
    //
    // The complete computation differs from bnorm and lnorm because of the way
    // scales and statistics are applied to the dst point. Since scales are
    // applied per channel, a part of derivative coming from the mean and
    // variance over different channels will be different, this is the primary
    // reason why accumulation is needed.
    //
    // y(c) = [(x(c) - m) / v] * gamma(c) + beta;
    // dy/dx = gamma(c) * [d(x(c) - m)/dx * v - (x(c) - m) * dv/dx] / v^2
    // `sum_dd_scaled` covers `d(x(c) - m)/dx` part;
    // `sum_dd_scaled_x_hat` covers a part coming from `dv/dx`;
    // The rest of values are computed at `Apply gradients` part.
    benchdnn_parallel_nd(MB, G, [&](int64_t mb, int64_t g) {
        int64_t stat_off = mb * G + g;
        float svar = var[stat_off];
        float rcp_denom = 1.f / sqrtf(svar + prb->eps);

        float sum_dd_scaled = 0.0f;
        float sum_dd_scaled_x_hat = 0.0f;

        if (!glob_stats) {
            for (int64_t c = prb->get_c_start(g); c < prb->get_c_start(g + 1);
                    ++c) {
                float gamma = use_sc ? sc[c] : 1.f;
                for_(int64_t d = 0; d < D; ++d)
                for_(int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    auto off = data_off(prb, mb, c, d, h, w);
                    float dd = d_dst[off];
                    float x_hat = src[off];

                    float dd_scaled = dd * gamma;
                    sum_dd_scaled += dd_scaled;
                    sum_dd_scaled_x_hat += dd_scaled * x_hat;
                }
            }
        }

        float mean_dd_scaled = sum_dd_scaled / CSP;
        float mean_dd_scaled_x_hat = sum_dd_scaled_x_hat / CSP;

        // Apply gradients
        for (int64_t c = prb->get_c_start(g); c < prb->get_c_start(g + 1);
                ++c) {
            float gamma = use_sc ? sc[c] : 1.f;
            for_(int64_t d = 0; d < D; ++d)
            for_(int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w) {
                auto off = data_off(prb, mb, c, d, h, w);
                float dd = d_dst[off];

                if (!glob_stats) {
                    float x_hat = src[off];
                    float dd_scaled = dd * gamma;
                    float ds = dd_scaled - mean_dd_scaled
                            - x_hat * mean_dd_scaled_x_hat;
                    d_src[off] = rcp_denom * ds;
                } else {
                    d_src[off] = dd * gamma * rcp_denom;
                }
            }
        }
    });
}

void compute_ref(const prb_t *prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref) {
    if (dir & FLAG_FWD)
        compute_ref_fwd(prb, args);
    else
        compute_ref_bwd(prb, args);
}

} // namespace gnorm
