/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "eltwise/eltwise.hpp"

namespace eltwise {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const auto src = args.find(DNNL_ARG_SRC).get_host_f32_handle();
    auto dst = args.find(DNNL_ARG_DST).get_host_f32_handle();

    const auto nelems = src.nelems();
    auto v_po_masks = prb->attr.post_ops.get_po_masks(prb->ndims);

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        float res
                = compute_eltwise_fwd(prb->alg, src[i], prb->alpha, prb->beta);

        const auto v_po_vals = prepare_po_vals(dst, args, v_po_masks, i);

        maybe_post_ops(prb->attr, res, 0.f, v_po_vals);

        // Backward use_dst case requires data adjustment since lower data type
        // may have less exact values which will be propagated further.
        res = ((prb->dir & FLAG_BWD) && prb->use_dst())
                ? round_to_nearest_representable(prb->dt, res)
                : res;
        dst[i] = res;
    });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    const auto src = args.find(DNNL_ARG_SRC).get_host_f32_handle();
    const auto dst = args.find(DNNL_ARG_DST).get_host_f32_handle();
    const auto source = prb->use_dst() ? dst : src;
    const auto d_dst = args.find(DNNL_ARG_DIFF_DST).get_host_f32_handle();
    auto d_src = args.find(DNNL_ARG_DIFF_SRC).get_host_f32_handle();

    const auto nelems = src.nelems();

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        d_src[i] = compute_eltwise_bwd(
                prb->alg, d_dst[i], source[i], prb->alpha, prb->beta);
    });
}

void compute_ref(const prb_t *prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref) {
    if (dir & FLAG_FWD)
        compute_ref_fwd(prb, args);
    else
        compute_ref_bwd(prb, args);
}

} // namespace eltwise
