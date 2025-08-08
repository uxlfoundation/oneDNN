/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include <assert.h>

#include "prelu/prelu.hpp"
#include "utils/parallel.hpp"

namespace prelu {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const auto src = args.find(DNNL_ARG_SRC).get_host_f32_handle();
    const auto wei = args.find(DNNL_ARG_WEIGHTS).get_host_f32_handle();
    auto dst = args.find(DNNL_ARG_DST).get_host_f32_handle();

    const auto nelems = src.nelems();
    const auto weights_broadcast_mask = prb->get_broadcast_mask();

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const auto wei_idx = src.get_idx(i, weights_broadcast_mask);
        const float s = src[i];
        float res = s * (s > 0 ? 1.f : wei[wei_idx]);
        maybe_saturate(prb->sdt[0], res);
        dst[i] = res;
    });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    const auto src = args.find(DNNL_ARG_SRC).get_host_f32_handle();
    const auto wei = args.find(DNNL_ARG_WEIGHTS).get_host_f32_handle();
    const auto d_dst = args.find(DNNL_ARG_DIFF_DST).get_host_f32_handle();
    auto d_src = args.find(DNNL_ARG_DIFF_SRC).get_host_f32_handle();
    auto d_wei = args.find(DNNL_ARG_DIFF_WEIGHTS).get_host_f32_handle();
    float *d_wei_buf = d_wei;

    const auto src_nelems = d_src.nelems();
    const auto wei_nelems = d_wei.nelems();

    const auto ker = [&](int64_t i, int64_t wei_idx, int64_t d_wei_idx) {
        float s = src[i];
        float dd = d_dst[i];
        float d_src_value = dd * (s > 0 ? 1.f : wei[wei_idx]);
        maybe_saturate(prb->sdt[0], d_src_value);
        d_src[i] = d_src_value;
        d_wei[d_wei_idx] += MIN2(0.f, s) * dd;
    };

    benchdnn_parallel_nd(wei_nelems, [&](int64_t i) { d_wei[i] = 0; });

    if (wei_nelems == 1) {
        const int reduce_dim = 0;
        const int64_t N = d_src.dims()[reduce_dim];
        const int64_t nelems_per_thr = src_nelems / N;
        d_wei_buf = new float[N];
        benchdnn_parallel_nd(N, [&](int64_t n) {
            d_wei_buf[n] = 0;

            for (int64_t ithr_i = 0; ithr_i < nelems_per_thr; ++ithr_i) {
                int64_t idx = nelems_per_thr * n + ithr_i;
                ker(idx, 0, n);
            }
        });

        for (int64_t i = 0; i < N; i++)
            d_wei[0] += d_wei_buf[i];
        delete[] d_wei_buf;

    } else if (src_nelems == wei_nelems) {
        benchdnn_parallel_nd(src_nelems, [&](int64_t i) { ker(i, i, i); });
    } else {
        const int64_t reduce_size = src_nelems / wei_nelems;

        // Re-used from ref_reduction.cpp
        // TODO: make a common reduction kernel to avoid duplication.
        const auto &src_dims = prb->vdims[0];
        const auto &wei_dims = prb->vdims[1];
        dims_t reduce_dims(prb->ndims, 1);
        for (int d = 0; d < prb->ndims; ++d)
            if (src_dims[d] != wei_dims[d]) reduce_dims[d] = src_dims[d];

        benchdnn_parallel_nd(wei_nelems, [&](int64_t f) {
            dims_t wei_pos = off2dims_idx(wei_dims, f);
            const int64_t wei_off = md_off_v(wei, wei_pos.data());
            const int64_t src_wei_off = md_off_v(src, wei_pos.data());

            for (int64_t r = 0; r < reduce_size; ++r) {
                dims_t reduce_pos = off2dims_idx(reduce_dims, r);
                const int64_t src_reduce_off = md_off_v(src, reduce_pos.data());
                const int64_t src_off = src_wei_off + src_reduce_off;
                ker(src_off, wei_off, wei_off);
            }
        });
    }
}

void compute_ref(const prb_t *prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref) {
    if (dir & FLAG_FWD)
        compute_ref_fwd(prb, args);
    else
        compute_ref_bwd(prb, args);
}

} // namespace prelu
