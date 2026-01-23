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

#if DNNL_EXPERIMENTAL_GROUPED_GEMM

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

    // src: [total_tokens, K] grouped
    // wei: [num_experts, K, N] dense with abc or acb
    // dst: [total_tokens, N] grouped
    const auto &src_grouped = src_d.sparse_desc().grouped_desc;
    const dim_t ngroups = src_grouped.ngroups;
    const dim_t K = wei_d.dims()[1];
    const dim_t N = wei_d.dims()[2];

    const void *src_data = CTX_IN_MEM(const void *, DNNL_ARG_SRC, 0);
    const int32_t *offsets = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
    const void *wei_data = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *dst_data = CTX_OUT_MEM(void *, DNNL_ARG_DST, 0);

    const auto src_dt = src_d.data_type();
    const auto wei_dt = wei_d.data_type();
    const auto dst_dt = pd()->dst_md()->data_type;

    const bool with_bias = pd()->with_bias();
    const void *bias_data
            = with_bias ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS) : nullptr;
    const auto bia_dt
            = with_bias ? pd()->weights_md(1)->data_type : data_type::undef;

    // src scales: row-wise
    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const void *src_scales = nullptr;
    if (with_src_scales) {
        src_scales
                = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    }

    // wei scales: column-wise or blocked (K grouping)
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_scale_dt = with_wei_scales
            ? attr_scales.get_data_type(DNNL_ARG_WEIGHTS)
            : data_type::undef;
    const auto wei_scale_group_k
            = with_wei_scales ? attr_scales.get_group(DNNL_ARG_WEIGHTS, 0) : 1;
    const auto wei_scale_ngroups_k
            = wei_scale_group_k > 1 ? K / wei_scale_group_k : 1;
    const void *wei_scales = nullptr;
    if (with_wei_scales) {
        wei_scales = CTX_IN_MEM(
                const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    }

    // Check if using int arithmetic for in4/int8 src/wei
    // Note: not for weights only quantization
    const bool use_int_arithmetic
            = utils::one_of(src_dt, data_type::s8, data_type::u8)
            && utils::one_of(wei_dt, data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4);

    // Parallelize over groups (experts in MoE)
    // Expectation is to see 128-256+ groups, with varying M per group
    // and possibly some empty groups (M == 0)
    parallel_nd(ngroups, [&](dim_t group_id) {
        const dim_t offset_start = (group_id == 0) ? 0 : offsets[group_id - 1];
        const dim_t offset_end = offsets[group_id];
        const dim_t M = offset_end - offset_start;

        if (M == 0) return; // skip if no rows in this group

        const dim_t src_base_idx = offset_start * K;
        const dim_t dst_base_idx = offset_start * N;

        for (dim_t m = 0; m < M; ++m) {
            for (dim_t n = 0; n < N; ++n) {
                float result = 0.0f;

                // For int, accumulate in int32 first, then apply scales
                // (as in ref_matmul_int8 kernel)
                if (use_int_arithmetic) {
                    for (dim_t i_group = 0; i_group < wei_scale_ngroups_k;
                            i_group++) {
                        const dim_t group_k = K / wei_scale_ngroups_k;
                        int acc = 0;

                        for (dim_t k = 0; k < group_k; ++k) {
                            const dim_t k_abs = k + i_group * group_k;
                            const dim_t src_idx = src_base_idx + m * K + k_abs;

                            dims_t wei_dims = {group_id, k_abs, n};
                            const dim_t wei_idx = wei_d.off_v(wei_dims);

                            const int s = io::load_int_value(
                                    src_dt, src_data, src_idx);
                            const int w = io::load_int_value(
                                    wei_dt, wei_data, wei_idx);
                            acc += s * w;
                        }

                        float acc_f = static_cast<float>(acc);

                        if (with_src_scales) {
                            const dim_t idx = offset_start + m;
                            const float src_scale = io::load_float_value(
                                    data_type::f32, src_scales, idx);
                            acc_f *= src_scale;
                        }

                        if (with_wei_scales) {
                            const dim_t idx = group_id * wei_scale_ngroups_k * N
                                    + i_group * N + n;
                            const float wei_scale = io::load_float_value(
                                    wei_scale_dt, wei_scales, idx);
                            acc_f *= wei_scale;
                        }

                        result += acc_f;
                    }
                } else {
                    // fp arithmetic
                    float acc = 0.0f;

                    for (dim_t k = 0; k < K; ++k) {
                        const dim_t src_idx = src_base_idx + m * K + k;

                        dims_t wei_dims = {group_id, k, n};
                        const dim_t wei_idx = wei_d.off_v(wei_dims);

                        const float s = io::load_float_value(
                                src_dt, src_data, src_idx);
                        const float w = io::load_float_value(
                                wei_dt, wei_data, wei_idx);
                        acc += s * w;
                    }

                    if (with_src_scales) {
                        const dim_t idx = offset_start + m;
                        const float scale = io::load_float_value(
                                data_type::f32, src_scales, idx);
                        acc *= scale;
                    }

                    if (with_wei_scales) {
                        const dim_t idx = group_id * N + n;
                        const float scale = io::load_float_value(
                                wei_scale_dt, wei_scales, idx);
                        acc *= scale;
                    }

                    result = acc;
                }

                // Add bias
                if (with_bias) {
                    const dim_t bias_idx = group_id * N + n;
                    result += io::load_float_value(bia_dt, bias_data, bias_idx);
                }

                const dim_t dst_idx = dst_base_idx + m * N + n;
                io::store_float_value(dst_dt, result, dst_data, dst_idx);
            }
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_GEMM
