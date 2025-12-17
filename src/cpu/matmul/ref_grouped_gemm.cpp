/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "cpu/matmul/ref_grouped_gemm.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

status_t ref_grouped_gemm_t::execute(const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));

    const auto &src_grouped = src_d.sparse_desc().grouped_desc;
    const dim_t num_groups = src_grouped.ngroups;

    // src: [total_tokens, K] grouped
    // wei: [num_experts, K, N] dense
    // dst: [total_tokens, N] grouped
    const dim_t K = src_d.dims()[1];
    const dim_t N = wei_d.dims()[2];

    const void *src_data = CTX_IN_MEM(const void *, DNNL_ARG_SRC, 0);
    const void *wei_data = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *dst_data = CTX_OUT_MEM(void *, DNNL_ARG_DST, 0);
    const int32_t *offsets = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);

    const auto src_dt = src_d.data_type();
    const auto wei_dt = wei_d.data_type();
    const auto dst_dt = pd()->dst_md()->data_type;

    const bool with_bias = pd()->with_bias();
    const void *bias_data
            = with_bias ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS) : nullptr;
    const auto bia_dt
            = with_bias ? pd()->weights_md(1)->data_type : data_type::undef;

    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const void *src_scales = nullptr;
    if (with_src_scales) {
        src_scales
                = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    }

    for (int group_id = 0; group_id < num_groups; ++group_id) {
        const dim_t M = offsets[group_id + 1] - offsets[group_id];
        if (M == 0) continue;

        const dim_t src_base_idx = offsets[group_id] * K;
        const dim_t dst_base_idx = offsets[group_id] * N;
        const dim_t wei_base_idx = group_id * K * N;

        for (dim_t m = 0; m < M; ++m) {
            for (dim_t n = 0; n < N; ++n) {
                float acc = 0.0f;

                for (dim_t k = 0; k < K; ++k) {
                    const dim_t src_idx = src_base_idx + m * K + k;
                    const dim_t wei_idx = wei_base_idx + k * N + n;

                    const float s
                            = io::load_float_value(src_dt, src_data, src_idx);
                    const float w
                            = io::load_float_value(wei_dt, wei_data, wei_idx);
                    acc += s * w;
                }

                if (with_src_scales) {
                    const dim_t token_idx = offsets[group_id] + m;
                    const float scale = io::load_float_value(
                            data_type::f32, src_scales, token_idx);
                    acc *= scale;
                }

                if (with_bias) {
                    const dim_t bias_idx = group_id * N + n;
                    acc += io::load_float_value(bia_dt, bias_data, bias_idx);
                }

                const dim_t dst_idx = dst_base_idx + m * N + n;
                io::store_float_value(dst_dt, acc, dst_data, dst_idx);
            }
        }
    }

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
