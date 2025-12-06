/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "cpu/gemm/gemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

status_t ref_grouped_gemm_t::execute(const exec_ctx_t &ctx) const {
    using namespace data_type;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto &src_grouped = src_d.sparse_desc().grouped_desc;

    const dim_t num_groups = src_grouped.ngroups;

    // src: [total_tokens, K] grouped
    // wei: [num_experts, K, N] dense 3D
    // dst: [total_tokens, N] grouped
    const dim_t K = src_d.dims()[1];
    const dim_t N = wei_d.dims()[2];

    // Buffer 0: values (concatenated data)
    const float *src_data = CTX_IN_MEM(const float *, DNNL_ARG_SRC, 0);
    const float *wei_data = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    float *dst_data = CTX_OUT_MEM(float *, DNNL_ARG_DST, 0);

    // Buffer 1: offsets array (cumulative boundaries [0, M0, M0+M1, ...])
    const int32_t *src_offsets = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
    const int32_t *dst_offsets = CTX_OUT_MEM(const int32_t *, DNNL_ARG_DST, 1);

    const bool with_bias = pd()->with_bias();
    const float *bias_data
            = with_bias ? CTX_IN_MEM(const float *, DNNL_ARG_BIAS) : nullptr;

    // Process each group
    for (int group_id = 0; group_id < num_groups; ++group_id) {
        // Calculate M for this group from offsets and skip empty groups
        const int M = src_offsets[group_id + 1] - src_offsets[group_id];
        if (M == 0) continue;

        const float *src_group = src_data + src_offsets[group_id] * K;
        float *dst_group = dst_data + dst_offsets[group_id] * N;
        const float *wei_group = wei_data + group_id * K * N;
        const float *bias_group
                = with_bias ? (bias_data + group_id * N) : nullptr;

        // Call GEMM for this group: dst_group = src_group * wei_data
        // TODO: bias is added manually since extended_sgemm seem to expect different layout(?)
        const dim_t lda = K;
        const dim_t ldb = N;
        const dim_t ldc = N;
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const dim_t M_dim = M;
        const dim_t N_dim = N;
        const dim_t K_dim = K;

        status_t status = extended_sgemm("N", "N", &N_dim, &M_dim, &K_dim,
                &alpha, wei_group, &ldb, src_group, &lda, &beta, dst_group,
                &ldc, nullptr, false);

        if (status != status::success) return status;

        if (with_bias) {
            for (dim_t m = 0; m < M; ++m) {
                for (dim_t n = 0; n < N; ++n) {
                    dst_group[m * N + n] += bias_group[n];
                }
            }
        }
    }

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
